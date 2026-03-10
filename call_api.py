import os
import requests
from pathlib import Path

API_URL = "http://localhost:8000/generate"
TEST_DIR = Path(__file__).parent / "test"
OUTPUT_DIR = Path(__file__).parent / "test_output"

OUTPUT_DIR.mkdir(exist_ok=True)

image_files = sorted(TEST_DIR.glob("*.jpg")) + sorted(TEST_DIR.glob("*.png"))

print(f"Found {len(image_files)} images")

for img_path in image_files:
    output_path = OUTPUT_DIR / f"{img_path.stem}.sog"
    print(f"Processing {img_path.name}...")

    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        data = {
            "auto_crop": True,
            "guidance_scale": 3.0,
            "random_seed": 4,
            "num_steps": 50,
        }
        response = requests.post(API_URL, files=files, data=data)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"  -> Saved {output_path}")
    else:
        print(f"  -> Error: {response.status_code} - {response.text}")

print("Done!")
