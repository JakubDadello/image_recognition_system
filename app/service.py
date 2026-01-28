import numpy as np
import bentoml
from bentoml.io import Image, JSON
from PIL import Image as PILImage
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Configuration ---
# CLASSES are defined here for global access
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# --- Initialize BentoML Runner ---
# This is the industry standard: separating the API logic from model inference
try:
    # We fetch the model from the local BentoML store
    model_runner = bentoml.tensorflow.get("resnet_custom_model:latest").to_runner()
except Exception as e:
    logger.error(f"Failed to initialize model runner: {e}")
    raise

# Define the service and its runners
service = bentoml.Service("industrial_defect_detector", runners=[model_runner])

@service.api(input=Image(), output=JSON())
async def predict_defect(img: PILImage.Image) -> dict:
    """
    Asynchronously processes the uploaded image and returns defect classification.
    """
    try:
        # 1. Preprocessing: Standardize image size and color space
        # Note the double parentheses in resize((224, 224))
        img = img.convert("RGB").resize((224, 224))
        
        # 2. Normalization: Scale pixels to [0, 1] range as expected by ResNet
        img_array = np.array(img) / 255.0
        
        # 3. Reshaping: Add batch dimension (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        # 4. Inference: Run the prediction using the dedicated runner
        preds = await model_runner.predict.async_run(img_array)
        
        # 5. Post-processing: Extract class name and confidence score
        class_index = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        logger.info(f"Prediction successful: {CLASSES[class_index]} ({confidence:.2f})")

        return {
            "defect_type": CLASSES[class_index],
            "confidence": round(confidence, 4),
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return {"status": "error", "message": str(e)}

@service.api(input=JSON(), output=JSON())
def health_check(input_data: dict) -> dict:
    """Standard health check endpoint for monitoring systems."""
    return {"status": "healthy", "service": "industrial-ai"}



