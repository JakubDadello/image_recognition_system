import os
import shutil
import pathlib
import tensorflow as tf
import bentoml
from tensorflow import keras

# --- Configuration ---
BASE_DIR = pathlib.Path(__file__).parent.absolute()
# Pointer to your raw weights file
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "steel_model_weights.h5") 
EXPORT_DIR = os.path.join(BASE_DIR, "temp_production_model")
MODEL_NAME = "resnet50_steel_defect"

IMG_SIZE = (200, 200)
NUM_CLASSES = 6

def build_production_model():
    """
    Reconstructs the model architecture exactly as used during training.
    Includes the preprocessing layer for an 'end-to-end' production graph.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_pixels")
    
    # Internal preprocessing (crucial for production)
    x = keras.applications.resnet50.preprocess_input(inputs)
    
    # Pretrained Backbone (weights=None because we load our own)
    base_model = keras.applications.ResNet50(
        include_top=False, 
        weights=None, 
        input_shape=(*IMG_SIZE, 3)
    )
    
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = keras.layers.Dropout(0.3, name="top_dropout")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    
    return keras.Model(inputs, outputs, name="Steel_Defect_Detector")

def migrate():
    try:
        # 1. Clean environment
        print("[*] Resetting Keras session...")
        tf.keras.backend.clear_session()

        # 2. Reconstruct architecture
        print("[*] Building clean model architecture...")
        model = build_production_model()

        # 3. Inject RAW weights only
        # This is the "safe" part that bypasses Keras 3 metadata errors
        print(f"[*] Loading raw weights from: {WEIGHTS_PATH}...")
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights file not found at {WEIGHTS_PATH}")
            
        model.load_weights(WEIGHTS_PATH)
        print("[+] Weights successfully injected into the architecture.")

        # 4. Export to SavedModel format (Standardized format for BentoML)
        if os.path.exists(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)
        
        print(f"[*] Exporting clean graph to: {EXPORT_DIR}...")
        # .export() creates a clean, framework-agnostic version of your model
        model.export(EXPORT_DIR) 

        # 5. BentoML Registration 
        print(f"[*] Registering in BentoML Model Store...")
        
        # Load the newly exported clean SavedModel
        reloaded_model = tf.saved_model.load(EXPORT_DIR)

        bento_model = bentoml.tensorflow.save_model(
            MODEL_NAME,
            reloaded_model, 
            # Note: we use "serve" as the default signature for exported models
            signatures={"serve": {"batchable": True, "batch_dim": 0}},
            metadata={
                "source_weights": os.path.basename(WEIGHTS_PATH),
                "input_shape": f"{IMG_SIZE}x3",
                "classes": NUM_CLASSES,
                "note": "Loaded from raw weights to ensure Keras 3 compatibility"
            }
        )
        print(f"[+] SUCCESS: Model registered as {bento_model.tag}")

    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {str(e)}")
        print("[*] Hint: Ensure you are running this from the project root folder.")

if __name__ == "__main__":
    migrate()