import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from prepare_raw_data import X, Y
from prepare_dataset import data_preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# =========================
# DATA PREPARATION
# =========================
# Perform full preprocessing pipeline:
# - train/val/test split
# - resizing, normalization
# - data augmentation (train only)
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


# =========================
# OPTIMIZER CONFIGURATION
# =========================
# Adam optimizer with fixed learning rate for baseline comparison
optimizer = keras.optimizers.Adam(learning_rate=0.001)


# =========================
# BASE MODEL (TRANSFER LEARNING)
# =========================
# Load pretrained ResNet50 without the classification head
base_model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

# Freeze all convolutional layers to use ResNet as a fixed feature extractor
base_model.trainable = False


# =========================
# CLASSIFICATION HEAD
# =========================
# Custom classification head on top of the pretrained backbone
inputs = base_model.input
x = base_model.output

# Global Average Pooling reduces spatial dimensions
x = keras.layers.GlobalAveragePooling2D()(x)

# Dropout for regularization
x = keras.layers.Dropout(0.3)(x)

# Final classification layer
outputs = keras.layers.Dense(101, activation="softmax")(x)

# Build the full model
model = keras.models.Model(inputs=inputs, outputs=outputs)


# =========================
# MODEL COMPILATION
# =========================
model.compile(
    loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"]
)


# =========================
# TRAINING
# =========================
# Train only the classification head while keeping the backbone frozen
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator 
)


# =========================
# EVALUATION
# =========================
# Evaluate model performance on unseen test data
model.evaluate(X_test, Y_test)

# Display model architecture summary
model.summary()


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
