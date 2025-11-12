import sys
import os
sys.path.append(os.path.abspath(".."))
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import models, layers, optimizers
from src.data_loader import X,Y
from src.preprocessor import data_preprocessing

model_path = "model_CNN.h5"

#definition of residual_unit
class ResidualUnit (keras.layers.Layer):
        def __init__ (self, filters, strides=1, activation='relu', **kwargs):
            super().__init__(**kwargs)
            self.activation = keras.layers.Activation(activation)
            self.main_layers = [
                keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False), 
                keras.layers.BatchNormalization(),
                keras.layers.Activation(activation), 
                keras.layers.Conv2D(filters, 3, strides = 1, padding = 'same', use_bias=False),
                keras.layers.BatchNormalization()]
            self.skip_layers = []
            if strides > 1:
                self.skip_layers = [
                    keras.layers.Conv2D(filters, 1, strides = strides, padding = 'same', use_bias = False),
                    keras.layers.BatchNormalization() ]
        def call(self, inputs):
            Z = inputs
            for layer in self.main_layers:
                Z = layer(Z)
            skip_Z = inputs 
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)
            return self.activation(Z + skip_Z) 

if os.path.exists(model_path):
    model = keras.models.load_model("model_CNN.h5", custom_objects={'ResidualUnit': ResidualUnit})
    print('Model loaded successfully')

else: 
    generator, val_data, test_data = data_preprocessing(X,Y) 
    X_val, Y_val = val_data
    X_test, Y_test = test_data

    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    #definition of sequential model 
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, input_shape = [224, 224, 3], padding ="same" , use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides = 2, padding = "same"))
    prev_filters = 64

    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2 
        model.add(ResidualUnit (filters, strides=strides))
        prev_filters = filters

    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(101, activation = "softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(generator, epochs=5, validation_data = (X_val, Y_val) ) 

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    model.summary()
    model.save(model_path)

    plt.plot(train_acc, label="Training")
    plt.plot(val_acc, label = "Validation")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.title("Learning curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluation")
    plt.show()



    

















