from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import torch
from PIL import Image
import io

app = FastAPI()

# Assuming the model and device setup is similar to the provided snippet.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

@app.post("/transform/")
async def transform_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_format = "png"  # Set the image format

    im_in = Image.open(io.BytesIO(contents)).convert("RGB")
    im_out = face2paint(model, im_in, side_by_side=False)

    buf = io.BytesIO()
    im_out.save(buf, format=image_format)
    buf.seek(0)

    return StreamingResponse(buf, media_type=f"image/{image_format}")
