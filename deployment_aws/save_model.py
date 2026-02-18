import os
import shutil
import pathlib
import tensorflow as tf
import bentoml
from tensorflow import keras

# -----------------------------------------------------------------------------
# PRODUCTION MODEL MIGRATION SCRIPT (Keras 3 + BentoML)
# Project: Steel Defect Detection - ResNet50 Transfer Learning
# Goal: Convert legacy .h5 weights into a clean BentoML SavedModel
# -----------------------------------------------------------------------------

# --- Configuration ---
BASE_DIR = pathlib.Path(__file__).parent.absolute()
WEIGHTS_PATH = os.path.join(BASE_DIR, "../","models", "resnet50_pretrained_best.h5")
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
    
    # Internal preprocessing - this was causing the 'Stack/Ellipsis' errors in .h5
    x = keras.applications.resnet50.preprocess_input(inputs)
    
    # Pretrained Backbone
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

        # 3. Inject weights
        # We use by_name=True to ensure weights map to the correct layers
        print(f"[*] Loading weights from: {WEIGHTS_PATH}...")
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
            
        model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print("[+] Weights successfully injected.")

        # 4. Intermediate Export (Fixes _DictWrapper & Keras 3 compatibility)
        # Saving without an extension in Keras 3 creates a clean SavedModel folder
        if os.path.exists(EXPORT_DIR):
            shutil.rmtree(EXPORT_DIR)
        
        print(f"[*] Exporting clean graph using model.export() to: {EXPORT_DIR}...")
        
        # W Keras 3 to jest zalecana metoda dla SavedModel (produkcja)
        model.export(EXPORT_DIR) 

        # 4. BENTOML REGISTRATION
        # 4. BENTOML REGISTRATION
        print(f"[*] Reloading clean graph for BentoML registration...")
        
        reloaded_model = tf.saved_model.load(EXPORT_DIR)

        print(f"[*] Finalizing registration in BentoML Model Store...")
        bento_model = bentoml.tensorflow.save_model(
            MODEL_NAME,
            reloaded_model, 
            signatures={"__call__": {"batchable": True, "batch_dim": 0}},
            metadata={
                "training_src": "resnet50_pretrained_best.h5",
                "input_shape": f"{IMG_SIZE}x3",
                "classes": NUM_CLASSES
            }
        )

        # Optional: Cleanup
        # shutil.rmtree(EXPORT_DIR)

    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {str(e)}")
        print("[*] Hint: Ensure you are running this from the project root folder.")

if __name__ == "__main__":
    migrate()