import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from einops import rearrange
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from huggingface_hub import snapshot_download
from PIL import Image

from gslrm.model.gaussians_renderer import render_turntable, imageseq2video
from mvdiffusion.pipelines.pipeline_mvdiffusion_unclip import (
    StableUnCLIPImg2ImgPipeline,
)
from utils_folder.face_utils import preprocess_image, preprocess_image_without_cropping

app = FastAPI(title="FaceLift API")

HF_REPO_ID = "wlyu/OpenFaceLift"

THREEDGS_CONVERTER_PATH = Path(__file__).parent / "third_party" / "3dgsconverter"


def download_weights_from_hf() -> Path:
    workspace_dir = Path(__file__).parent

    mvdiffusion_path = workspace_dir / "checkpoints/mvdiffusion/pipeckpts"
    gslrm_path = workspace_dir / "checkpoints/gslrm/ckpt_0000000000021125.pt"
    prompt_embeds_path = (
        workspace_dir / "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"
    )

    if (
        mvdiffusion_path.exists()
        and gslrm_path.exists()
        and prompt_embeds_path.exists()
    ):
        print("Using local model weights")
        return workspace_dir

    print(f"Downloading model weights from HuggingFace: {HF_REPO_ID}")
    print("This may take a few minutes on first run...")

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=str(workspace_dir / "checkpoints"),
        local_dir_use_symlinks=False,
    )

    print("Model weights downloaded successfully!")
    return workspace_dir


class FaceLiftPipeline:
    def __init__(self):
        workspace_dir = download_weights_from_hf()

        self.output_dir = workspace_dir / "outputs"
        self.examples_dir = workspace_dir / "examples"
        self.output_dir.mkdir(exist_ok=True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = 512
        self.camera_indices = [2, 1, 0, 5, 4, 3]

        print("Loading models...")
        self.mvdiffusion_pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
            str(workspace_dir / "checkpoints/mvdiffusion/pipeckpts"),
            torch_dtype=torch.float16,
        )
        self.mvdiffusion_pipeline.unet.enable_xformers_memory_efficient_attention()
        self.mvdiffusion_pipeline.to(self.device)

        with open(workspace_dir / "configs/gslrm.yaml", "r") as f:
            config = edict(yaml.safe_load(f))

        module_name, class_name = config.model.class_name.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        ModelClass = getattr(module, class_name)

        self.gs_lrm_model = ModelClass(config)
        checkpoint = torch.load(
            workspace_dir / "checkpoints/gslrm/ckpt_0000000000021125.pt",
            map_location="cpu",
        )
        self.gs_lrm_model.load_state_dict(checkpoint["model"])
        self.gs_lrm_model.to(self.device)

        self.color_prompt_embedding = torch.load(
            workspace_dir / "mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt",
            map_location=self.device,
        )

        with open(workspace_dir / "utils_folder/opencv_cameras.json", "r") as f:
            self.cameras_data = json.load(f)["frames"]

        print("Models loaded successfully!")

    def generate_3d_head(
        self,
        image_path,
        auto_crop=True,
        guidance_scale=3.0,
        random_seed=4,
        num_steps=50,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.output_dir / timestamp
        output_dir.mkdir(exist_ok=True)

        original_img = np.array(Image.open(image_path))
        input_image = (
            preprocess_image(original_img)
            if auto_crop
            else preprocess_image_without_cropping(original_img)
        )

        if input_image.size != (self.image_size, self.image_size):
            input_image = input_image.resize((self.image_size, self.image_size))

        input_path = output_dir / "input.png"
        input_image.save(input_path)

        generator = torch.Generator(device=self.mvdiffusion_pipeline.unet.device)
        generator.manual_seed(random_seed)

        result = self.mvdiffusion_pipeline(
            input_image,
            None,
            prompt_embeds=self.color_prompt_embedding,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=num_steps,
            generator=generator,
            eta=1.0,
        )

        selected_views = result.images[:6]

        multiview_image = Image.new("RGB", (self.image_size * 6, self.image_size))
        for i, view in enumerate(selected_views):
            multiview_image.paste(view, (self.image_size * i, 0))

        multiview_path = output_dir / "multiview.png"
        multiview_image.save(multiview_path)

        view_arrays = [np.array(view) for view in selected_views]
        lrm_input = torch.from_numpy(np.stack(view_arrays, axis=0)).float()
        lrm_input = lrm_input[None].to(self.device) / 255.0
        lrm_input = rearrange(lrm_input, "b v h w c -> b v c h w")

        selected_cameras = [self.cameras_data[i] for i in self.camera_indices]
        fxfycxcy_list = [[c["fx"], c["fy"], c["cx"], c["cy"]] for c in selected_cameras]
        c2w_list = [np.linalg.inv(np.array(c["w2c"])) for c in selected_cameras]

        fxfycxcy = torch.from_numpy(np.stack(fxfycxcy_list, axis=0).astype(np.float32))
        c2w = torch.from_numpy(np.stack(c2w_list, axis=0).astype(np.float32))
        fxfycxcy = fxfycxcy[None].to(self.device)
        c2w = c2w[None].to(self.device)

        batch_indices = torch.stack(
            [
                torch.zeros(lrm_input.size(1)).long(),
                torch.arange(lrm_input.size(1)).long(),
            ],
            dim=-1,
        )[None].to(self.device)

        batch = edict(
            {
                "image": lrm_input,
                "c2w": c2w,
                "fxfycxcy": fxfycxcy,
                "index": batch_indices,
            }
        )

        with torch.autocast(enabled=True, device_type="cuda", dtype=torch.float16):
            result = self.gs_lrm_model.forward(
                batch, create_visual=False, split_data=True
            )

        comp_image = result.render[0].unsqueeze(0).detach()
        gaussians = result.gaussians[0]

        filtered_gaussians = gaussians.apply_all_filters(
            cam_origins=None,
            opacity_thres=0.04,
            scaling_thres=0.2,
            floater_thres=0.75,
            crop_bbx=[-0.91, 0.91, -0.91, 0.91, -0.6, 1.0],
            nearfar_percent=(0.0001, 1.0),
        )

        ply_path = output_dir / "gaussians.ply"
        filtered_gaussians.save_ply(str(ply_path))

        if not THREEDGS_CONVERTER_PATH.exists():
            raise RuntimeError(
                "3dgsconverter submodule is missing. Please initialize submodules first."
            )

        sog_path = output_dir / "gaussians.sog"
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        extra_pythonpath = str(THREEDGS_CONVERTER_PATH)
        env["PYTHONPATH"] = (
            f"{extra_pythonpath}:{current_pythonpath}"
            if current_pythonpath
            else extra_pythonpath
        )

        conversion_cmd = [
            sys.executable,
            "-m",
            "gsconverter.main",
            "-i",
            str(ply_path),
            "-o",
            str(sog_path),
            "-f",
            "sog",
            "--compression_level",
            "9",
            "--rgb",
            "--force",
        ]
        subprocess.run(conversion_cmd, check=True, env=env)

        return str(sog_path), str(output_dir)


pipeline = None


@app.on_event("startup")
async def startup_event():
    global pipeline
    pipeline = FaceLiftPipeline()


@app.post("/generate")
async def generate_3d(
    file: UploadFile = File(...),
    auto_crop: bool = Form(True),
    guidance_scale: float = Form(3.0),
    random_seed: int = Form(4),
    num_steps: int = Form(50),
):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        sog_path, output_dir = pipeline.generate_3d_head(
            tmp_path,
            auto_crop=auto_crop,
            guidance_scale=guidance_scale,
            random_seed=random_seed,
            num_steps=num_steps,
        )

        return FileResponse(
            sog_path,
            media_type="application/octet-stream",
            filename="face_3d.sog",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
