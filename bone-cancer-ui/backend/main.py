from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
from PIL import Image
from model_inference import get_prediction_and_cam
import base64

app = FastAPI(title="Bone Cancer Detection API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    message: str
    gradcam_base64: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    # Read and open the uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        # Run validation + inference + Grad-CAM
        class_name, confidence, cam_img = get_prediction_and_cam(image)

        # Convert Grad-CAM image to Base64 for the frontend
        buffered = io.BytesIO()
        cam_img.save(buffered, format="JPEG")
        cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return PredictionResponse(
            class_name=class_name,
            confidence=confidence,
            message="Successfully processed image.",
            gradcam_base64=cam_base64
        )

    except ValueError as e:
        # This is triggered when is_valid_xray() rejects the image.
        # Returns a clean error to the frontend instead of crashing.
        return PredictionResponse(
            class_name="Invalid Image",
            confidence=0.0,
            message=str(e),
            gradcam_base64=""
        )

    except Exception as e:
        # Catches any unexpected errors (model errors, file corruption, etc.)
        return PredictionResponse(
            class_name="Error",
            confidence=0.0,
            message=str(e),
            gradcam_base64=""
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)