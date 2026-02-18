import os
import shutil
import pathlib
import tensorflow as tf
from tensorflow import keras

# -----------------------------------------------------------------------------
# PRODUCTION MODEL MIGRATION SCRIPT (Keras 3 Native)
# Project: Steel Defect Detection - ResNet50 Transfer Learning
# Goal: Convert legacy .h5 weights into a stable .keras bundle
# -----------------------------------------------------------------------------

# --- Configuration ---
BASE_DIR = pathlib.Path(__file__).parent.absolute()
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "resnet50_pretrained_best.h5")
# We save to a single .keras file instead of a directory to avoid _DictWrapper errors
EXPORT_FILE = os.path.join(BASE_DIR, "steel_defect_model_final.keras")

IMG_SIZE = (200, 200)
NUM_CLASSES = 6

def build_production_model():
    """
    Reconstructs the model architecture exactly as used during training.
    Includes the preprocessing layer for an 'end-to-end' production graph.
    """
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_pixels")
    
    # --- Internal preprocessing (ResNet50 specific) ---
    x = keras.applications.resnet50.preprocess_input(inputs)
    
    # --- Pretrained Backbone ---
    # We use None for weights because we will load our custom .h5 weights later
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
        print(f"[*] Loading weights from: {WEIGHTS_PATH}...")
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")
            
        # Strategy: Load weights into the ResNet layer first to ensure backbone integrity
        # Then load into the whole model for the top 'Dense' layers
        try:
            # Finding the resnet layer by its class or name
            resnet_layer = model.get_layer("resnet50")
            resnet_layer.load_weights(WEIGHTS_PATH, by_name=True)
            print("[+] Backbone weights injected.")
        except Exception as e:
            print(f"[!] Warning: Could not target backbone directly: {e}")

        model.load_weights(WEIGHTS_PATH, by_name=True, skip_mismatch=True)
        print("[+] Final weight injection complete.")

        # 4. Save to Native Keras 3 format
        # This avoids the SavedModel/BentoML serialization that triggers _DictWrapper errors
        print(f"[*] Saving model to native Keras 3 format: {EXPORT_FILE}...")
        
        if os.path.exists(EXPORT_FILE):
            os.remove(EXPORT_FILE)
            
        model.save(EXPORT_FILE)

        print(f"\n[+++] SUCCESS! Upload '{os.path.basename(EXPORT_FILE)}' to Hugging Face.")

    except Exception as e:
        print(f"\n[!] CRITICAL ERROR: {str(e)}")
        print("[*] Hint: Ensure you are running this from the project root folder.")

if __name__ == "__main__":
    migrate()