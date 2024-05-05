# Liveness Detection API

A lightweight REST API serving a PyTorch-based liveness detection model, designed to prevent presentation attacks in facial recognition pipelines.

## Project Layout
- `api/` : FastAPI application and model inference class.
- `weights/` : Storage for `.pth` model weights.

## Usage
```sh
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
Then send a POST request with an image to `/detect_liveness`.
