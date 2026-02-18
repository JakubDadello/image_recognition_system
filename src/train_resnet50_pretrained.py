import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import logging

from data_pipeline import DataPipeline

# ------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------------------------------------------------
# Reproducibility (Seed)
# ------------------------------------------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
MODEL_PATH = "models/resnet50_pretrained.keras"
BEST_MODEL_PATH = "models/resnet50_pretrained_best.keras"
WEIGHTS_PATH = "models/steel_model_weights.weights.h5"
PLOT_PATH = "reports/history_resnet50_pretrained.png" 

ZIP_PATH = "data/dataset.zip"
EXTRACT_PATH = "data/steel_data"

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6


# ------------------------------------------------------------
# ResNet-50 Builder
# ------------------------------------------------------------
def build_resnet50(input_shape, num_classes):
    """
    Builds a transfer learning model using a pretrained ResNet-50 backbone.
    
    This implementation uses the Keras Functional API to include preprocessing 
    directly within the model graph, ensuring portability for production 
    deployments (e.g., AWS SageMaker).
    """
    
    # --- Input Layer ---
    # Accepts raw pixel values (0-255) as defined in the DataPipeline
    inputs = keras.Input(shape=input_shape)

    # --- Preprocessing Layer ---
    # Specifically designed for ResNet50: 
    x = keras.applications.resnet50.preprocess_input(inputs)

    # --- Pretrained Backbone ---
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Freeze the convolutional base to prevent weights from being updated 
    # during the initial training phase (Transfer Learning).
    base_model.trainable = False

    # Pass the preprocessed input through the frozen backbone.
    # 'training=False' ensures BatchNormalization layers run in inference mode.
    x = base_model(x, training=False) 

    # --- Classification Head ---
    # Convert 4D feature maps to a 2D feature vector.
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Regularization layer to prevent overfitting.
    x = keras.layers.Dropout(0.3)(x)
    
    # Final output layer with Softmax activation for multi-class classification.
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    # Instantiate the final Keras Model.
    model = keras.Model(inputs, outputs, name="ResNet50_TransferLearning_Production")
    
    return model

# ------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------
if __name__ == "__main__":
    logging.info("Initializing data pipeline...")

    pipeline = DataPipeline(
        ZIP_PATH,
        EXTRACT_PATH,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    train_data, val_data, test_data = pipeline.running_engine()

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------
    if os.path.exists(MODEL_PATH):
        logging.info(f"Loading existing model from {MODEL_PATH}...")
        model = keras.models.load_model(MODEL_PATH)
    else:
        logging.info("Creating new ResNet-50 transfer learning model...")
        model = build_resnet50(
            input_shape=(200, 200, 3),
            num_classes=NUM_CLASSES
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # --------------------------------------------------------
    # Callbacks (Production-Ready)
    # --------------------------------------------------------
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    logging.info("Starting training...")
    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=val_data,
        callbacks=callbacks
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------
    logging.info("Evaluating on test set...")
    loss, accuracy = model.evaluate(test_data)
    logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_PATH)
    model.save_weights(WEIGHTS_PATH)
    logging.info(f"Final model saved to {MODEL_PATH}")

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy", color="tab:blue")
    ax1.plot(history.history["accuracy"], label="Train Acc", color="tab:blue", linewidth=2)
    ax1.plot(history.history["val_accuracy"], label="Val Acc", color="tab:cyan", linestyle="--")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="tab:red")
    ax2.plot(history.history["loss"], label="Train Loss", color="tab:red", alpha=0.5)
    ax2.plot(history.history["val_loss"], label="Val Loss", color="tab:orange", linestyle="--")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    os.makedirs("reports", exist_ok=True)

    plt.title("Steel Defect Detection: ResNet-50 Transfer Learning (Production)")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOT_PATH)
    logging.info(f"Plot saved to {PLOT_PATH}")
    
    plt.show()