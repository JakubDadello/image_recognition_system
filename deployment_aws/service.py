import numpy as np
import bentoml
import cv2
import logging
from PIL import Image as PILImage

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
TARGET_IMG_SIZE = (200, 200)

@bentoml.service(
    name="industrial_defect_detector",
    traffic={"timeout": 60},
)
class IndustrialDefectService:
    # Model reference from local store
    model_ref = bentoml.models.get("resnet50_steel_defect:latest")

    def __init__(self):
        # Runner initialization
        self.runner = self.model_ref.to_runner()

    # CORRECTED DECORATOR: 
    # In newer BentoML, we don't pass 'input' or 'output' here.
    # The types are inferred from the function signature below.
    @bentoml.api
    async def predict_defect(self, img: PILImage.Image) -> dict:
        """
        Receives an image and returns classification results as a dictionary (JSON).
        """
        try:
            # 1. Image Conversion
            img_np = np.array(img.convert("RGB"))

            # 2. OpenCV Enhancement
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_res = cv2.resize(img_rgb, TARGET_IMG_SIZE)

            # 3. Input Preparation (No manual /255.0)
            img_array = img_res.astype(np.float32) 
            img_array = np.expand_dims(img_array, axis=0)

            # 4. Model Inference
            preds = await self.runner.run(img_array)
            
            # 5. Output Post-processing
            class_index = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            logger.info(f"Inference Successful: Class={CLASSES[class_index]}, Confidence={confidence:.4f}")

            return {
                "defect_type": CLASSES[class_index],
                "confidence": round(confidence, 4),
                "status": "success"
            }

        except Exception as e:
            logger.error(f"Inference pipeline failed: {str(e)}")
            return {"status": "error", "message": str(e)}

    @bentoml.api
    def health_check(self) -> dict:
        return {"status": "healthy", "service": "industrial-ai-gateway"}