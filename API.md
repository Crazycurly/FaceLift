# FaceLift API Documentation

Base URL: `http://localhost:8000`

## Endpoints

### Health Check

**GET** `/health`

Check if the API is running.

**Response:**
```json
{
  "status": "healthy"
}
```

---

### Generate 3D Head

**POST** `/generate`

Generate a 3D head model from an input image.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file` | File | Yes | - | Input image (PNG/JPG) |
| `auto_crop` | boolean | No | `true` | Whether to auto-crop the face |
| `guidance_scale` | float | No | `3.0` | Guidance scale for the diffusion model |
| `random_seed` | integer | No | `4` | Random seed for reproducibility |
| `num_steps` | integer | No | `50` | Number of inference steps |

**Example Request (curl):**
```bash
curl -X POST "http://localhost:8000/generate" \
  -F "file=@input.jpg" \
  -F "auto_crop=true" \
  -F "guidance_scale=3.0" \
  -F "random_seed=4" \
  -F "num_steps=50" \
  -o face_3d.sog
```

**Response:** Binary file (`.sog` - 3D Gaussian Splatting format)

**Status Codes:**
- `200`: Success - Returns the 3D model file
- `500`: Server error

---

## Running the Server

```bash
python fastapi_server.py
```

The server will start on `http://0.0.0.0:8000`