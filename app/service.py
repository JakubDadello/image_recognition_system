import numpy as np
import bentoml
import cv2
import logging
from bentoml.io import Image, JSON
from PIL import Image as PILImage

# --- Logging Configuration ---
# Standard production logging to track inference and potential errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Constants ---
# Defined to ensure consistency between training and deployment
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TARGET_IMG_SIZE = (224, 224)

# --- Initialize BentoML Runner ---
# We use the Runner API to scale the model inference independently from the API logic
try:
    # Model is fetched from the local BentoML store
    model_runner = bentoml.tensorflow.get("resnet50_model:latest").to_runner()
except Exception as e:
    logger.error(f"Failed to initialize model runner: {e}")
    raise

# --- Service Definition ---
# This object manages the runners and exposes the API endpoints
service = bentoml.Service("industrial_defect_detector", runners=[model_runner])

@service.api(input=Image(), output=JSON())
async def predict_defect(img: PILImage.Image) -> dict:
    """
    Inference Endpoint:
    Receives an image, performs Computer Vision preprocessing via OpenCV,
    and returns the classification results from the ResNet50 model.
    """
    try:
        # 1. Convert PIL Image to NumPy array (RGB) for OpenCV processing
        img_np = np.array(img.convert("RGB"))

        # 2. Computer Vision Enhancement (OpenCV)
        # Convert RGB to BGR as OpenCV standard, apply filters, then resize
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Optional: Noise reduction using Gaussian Blur
        img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)

        # Convert back to RGB and resize for the model input requirements
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, TARGET_IMG_SIZE)

        # 3. Normalization and Batch Dimension
        # Scaling pixel values to [0, 1] and expanding dimensions to (1, 224, 224, 3)
        img_array = img_res.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 4. Model Inference
        # Asynchronous execution to maximize performance under high load
        preds = await model_runner.predict.async_run(img_array)
        
        # 5. Output Post-processing
        # Mapping the numerical output to the corresponding defect class
        class_index = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        logger.info(f"Inference Successful: Class={CLASSES[class_index]}, Confidence={confidence:.4f}")

        return {
            "defect_type": CLASSES[class_index],
            "confidence": round(confidence, 4),
            "input_shape": list(img_array.shape),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Inference pipeline failed: {str(e)}")
        return {
            "status": "error",
            "message": "An error occurred during image processing or inference."
        }

@service.api(input=JSON(), output=JSON())
def health_check(input_data: dict) -> dict:
    """Standard health check endpoint for monitoring cloud instance status."""
    return {"status": "healthy", "service": "industrial-ai-gateway"}