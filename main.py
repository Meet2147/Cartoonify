

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from PIL import Image
import torch
import uuid
import os
import io

app = FastAPI()

# # Redis setup
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)  # Update host/port as per your setup

# JWT configuration
SECRET_KEY = "Mahantam#6556"  # Replace with a strong secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Dummy user database
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "hashed_password": pwd_context.hash("testpassword"),  # Replace with hashed password
    },
   
    "Sahil@6556": {
        "username": "Sahil@6556",
        "hashed_password": pwd_context.hash("Sahil#1225"),
    }
}


# Device setup for AnimeGAN model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load(
    "bryandlee/animegan2-pytorch:main",
    "generator",
    pretrained="face_paint_512_v2",
    device=device
).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)


# Helper functions for JWT and authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict):
    return jwt.encode(data, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")


# Token endpoint for login
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}


# # Transform image endpoint
# @app.post("/transform/")
# async def transform_image(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
#     contents = await file.read()
#     image_format = "png"  # Output image format

#     # Open and process the input image
#     im_in = Image.open(io.BytesIO(contents)).convert("RGB")
#     im_out = face2paint(model, im_in, side_by_side=False)

#     # Generate a unique filename and save to /tmp
#     output_filename = f"{uuid.uuid4()}.{image_format}"
#     output_path = f"/tmp/{output_filename}"  # /tmp for Lambda-like environments
#     im_out.save(output_path, format=image_format)

#     # Return the processed image
#     return FileResponse(output_path, media_type=f"image/{image_format}", filename=output_filename)

from fastapi.responses import StreamingResponse

@app.post("/transform/")
async def transform_image(file: UploadFile = File(...), current_user: str = Depends(get_current_user)):
    """
    Transform the uploaded image using the AnimeGAN model and return the transformed image.
    Requires a valid authenticated user.
    """
    # Log the current user (for debugging or tracking purposes)
    print(f"Transform request received from user: {current_user}")

    # Read the uploaded file
    contents = await file.read()
    image_format = "png"  # Set the image format

    # Open the input image and apply the transformation
    im_in = Image.open(io.BytesIO(contents)).convert("RGB")
    im_out = face2paint(model, im_in, side_by_side=False)

    # Save the transformed image to an in-memory buffer
    buf = io.BytesIO()
    im_out.save(buf, format=image_format)
    buf.seek(0)

    # Return the transformed image as a streaming response
    return StreamingResponse(buf, media_type=f"image/{image_format}")
