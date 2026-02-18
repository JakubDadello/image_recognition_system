import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as PILImage

MODEL_PATH =  "./temp_production_model/"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
IMG_SIZE = (200, 200)

model = tf.keras.Sequential([
    tf.keras.layers.TFSMLayer('./temp_production_model/', call_endpoint='serving_default')
])

# Keras 3 version (multisystem Keras API) 
# model = tf.keras.models.load_model(MODEL_PATH)

def predictions (img: PILImage.Image):

    # 1. Image Conversion
    img_np = np.array(img.convert("RGB"))

     # 2. OpenCV Enhancement
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    img_rgb= cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, IMG_SIZE )

    # 3. Input Preparation
    img_array = img_res.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Model Inference
    preds = model.predict(img_array)

    # 5. Output Post-processing
    probabilities = list(preds.values())[0][0]

    return {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

demo = gr.Interface(
    fn=predictions, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Label(num_top_classes=3)
    )
    
if __name__ == "__main__":
    demo.launch()