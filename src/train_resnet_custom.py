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
MODEL_PATH = "models/resnet_custom.keras"
BEST_MODEL_PATH = "models/resnet_custom_best.keras"
PLOT_PATH = "reports/history_resnet_custom.png" 

ZIP_PATH = "data/dataset.zip"
EXTRACT_PATH = "data/steel_data"

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 6


# ------------------------------------------------------------
# Residual Block
# ------------------------------------------------------------
class ResidualBlock(keras.layers.Layer):
    """
    Custom residual block implementing skip connections as used in ResNet architectures.
    """

    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)

        self.activation = keras.layers.Activation(activation)

        # Main convolutional path
        self.main_layers = keras.Sequential([
            keras.layers.Conv2D(
                filters, kernel_size=3, strides=strides,
                padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
            keras.layers.Conv2D(
                filters, kernel_size=3, strides=1,
                padding="same", use_bias=False
            ),
            keras.layers.BatchNormalization()
        ])

        # Skip connection (projection shortcut if dimensions change)
        if strides > 1:
            self.skip_layers = keras.Sequential([
                keras.layers.Conv2D(
                    filters, kernel_size=1, strides=strides,
                    padding="same", use_bias=False
                ),
                keras.layers.BatchNormalization()
            ])
        else:
            self.skip_layers = keras.layers.Activation("linear")

    def call(self, inputs):
        x = self.main_layers(inputs)
        skip = self.skip_layers(inputs)
        return self.activation(x + skip)


# ------------------------------------------------------------
# Custom ResNet Builder
# ------------------------------------------------------------
def build_custom_resnet(input_shape, num_classes):
    """
    Builds a ResNet-34 style architecture using the custom ResidualBlock.
    """
    inputs = keras.Input(shape=input_shape)

    # Stem
    x = keras.layers.Conv2D(
        64, kernel_size=7, strides=2, padding="same", use_bias=False
    )(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # ResNet-34 configuration
    res_config = [
        (64, 3),
        (128, 4),
        (256, 6),
        (512, 3),
    ]

    prev_filters = 64
    for filters, blocks in res_config:
        for block_idx in range(blocks):
            strides = 2 if block_idx == 0 and filters != prev_filters else 1
            x = ResidualBlock(filters, strides=strides)(x)
            prev_filters = filters

    # Classification head
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="Custom_ResNet_Steel")


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
        model = keras.models.load_model(
            MODEL_PATH,
            custom_objects={"ResidualBlock": ResidualBlock}
        )
    else:
        logging.info("Creating new Custom ResNet model...")
        model = build_custom_resnet(
            input_shape=(*IMG_SIZE, 3),
            num_classes=NUM_CLASSES
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
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

    plt.title("Steel Defect Detection: Custom ResNet Training History")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
 
    plt.savefig(PLOT_PATH)
    logging.info(f"Plot saved to {PLOT_PATH}")

    plt.show()
