import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import models, layers, optimizers
from prepare_raw_data import X,Y
from prepare_dataset import data_preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = "model_CNN.h5"


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
                filters,
                kernel_size=3,
                strides=strides,
                padding="same",
                use_bias=False
            ),
            keras.layers.BatchNormalization(),
            keras.layers.Activation(activation),
            keras.layers.Conv2D(
                filters,
                kernel_size=3,
                strides=1,
                padding="same",
                use_bias=False
            ),
            keras.layers.BatchNormalization()
        ])

        # Skip connection
        # Projection shortcut is used when spatial dimensions change
        if strides > 1:
            self.skip_layers = keras.Sequential([
                keras.layers.Conv2D(
                    filters,
                    kernel_size=1,
                    strides=strides,
                    padding="same",
                    use_bias=False
                ),
                keras.layers.BatchNormalization()
            ])
        else:
            # Identity mapping
            self.skip_layers = keras.layers.Activation("linear")

    def call(self, inputs):
        """
        Forward pass for the residual block.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying residual connection.
        """
        x = self.main_layers(inputs)
        skip = self.skip_layers(inputs)

        # Element-wise addition of main path and skip connection
        return self.activation(x + skip)

   
# =========================
# MODEL LOADING OR TRAINING
# =========================

if os.path.exists(model_path):
    # Load previously trained model with custom residual blocks
    model = keras.models.load_model(
        model_path,
        custom_objects={"ResidualBlock": ResidualBlock}
    )
    print("Model loaded successfully")

else:
    # =========================
    # DATA PREPARATION
    # =========================
    train_generator, val_data, test_data = data_preprocessing(X, Y)

    X_val, Y_val = val_data
    X_test, Y_test = test_data

    val_data_gen = ImageDataGenerator()  

    val_generator = val_data_gen.flow(
        X_val,
        Y_val,
        batch_size=32,
        shuffle=False
    )


    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # =========================
    # MODEL DEFINITION
    # =========================
    model = keras.models.Sequential()

    # Initial convolution block (as in original ResNet)
    model.add(
        keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            strides=2,
            padding="same",
            use_bias=False,
            input_shape=(224, 224, 3)
        )
    )
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"))

    # =========================
    # RESIDUAL BLOCK STACK
    # ResNet-34 style: 3-4-6-3
    # =========================
    prev_filters = 64

    Res_config = [
    (64, 3),
    (128, 4),
    (256, 6),
    (512, 3),
    ]
    
    for filters, blocks in Res_config:
        for block in range(blocks):
            strides = 2 if block == 0 and filters != prev_filters else 1
            model.add(ResidualBlock(filters, strides=strides))
            prev_filters = filters

    # =========================
    # CLASSIFICATION HEAD
    # =========================
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(101, activation="softmax"))

    # =========================
    # COMPILATION
    # =========================
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # =========================
    # TRAINING
    # =========================
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator 
    )

    # =========================
    # EVALUATION
    # =========================
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    model.summary()
    model.save(model_path)

    # =========================
    # LEARNING CURVES
    # =========================
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(history.history["accuracy"], label="Train Acc", color='tab:blue')
    ax1.plot(history.history["val_accuracy"], label="Val Acc", color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(history.history["loss"], label="Train Loss", color='tab:red')
    ax2.plot(history.history["val_loss"], label="Val Loss", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Learning Curve – ResNet50 Baseline")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.show()
