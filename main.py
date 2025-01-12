from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch
from PIL import Image
import io
import os
import uuid
from mangum import Mangum

app = FastAPI()
# Mangum adapter for Vercel
handler = Mangum(app)

# Assuming the model and device setup is similar to the provided snippet.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

@app.post("/transform/")
async def transform_image(file: UploadFile = File(...)):
    contents = await file.read()
    image_format = "png"  # Set the image format

    # Open the input image
    im_in = Image.open(io.BytesIO(contents)).convert("RGB")
    im_out = face2paint(model, im_in, side_by_side=False)

    # Generate a unique filename to save the transformed image
    output_filename = f"{uuid.uuid4()}.{image_format}"
    output_path = f"/tmp/{output_filename}"  # Save in /tmp for Lambda-like environments

    # Save the transformed image
    im_out.save(output_path, format=image_format)

    # Return the file as a downloadable response
    return FileResponse(output_path, media_type=f"image/{image_format}", filename=output_filename)

