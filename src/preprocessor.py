import numpy as np
import tensorflow as tf
from src.data_loader import X,Y
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_preprocessing (X,Y, IMG_SIZE=224, test_size=0.20, val_size=0.25, batch_size=32, num_classes=101):

    #data splitting
    X_rest, X_test, Y_rest, Y_test = train_test_split(X,Y, test_size=test_size, random_state=42, stratify=Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X_rest, Y_rest, test_size=val_size, random_state=42, stratify=Y_rest)

    #data resizing
    X_train = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy() for img in X_train])
    X_val = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy() for img in X_val])
    X_test = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy() for img in X_test])


    #data normalization (data scaling)
    X_train = X_train.astype('float32')/255.0
    X_val = X_val.astype('float32')/255.0
    X_test = X_test.astype('float32')/255.0

    #data transformation 
    Y_train = np.array(Y_train) - 1
    Y_val   = np.array(Y_val) - 1
    Y_test  = np.array(Y_test) - 1  

    Y_train= to_categorical(Y_train, num_classes = num_classes)
    Y_val  = to_categorical(Y_val, num_classes= num_classes)
    Y_test = to_categorical(Y_test, num_classes = num_classes)

    #data augmentation
    datagen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True)
    
    datagen.fit(X_train)
    generator = datagen.flow(X_train, Y_train, batch_size=batch_size)

    
    return generator, (X_val, Y_val), (X_test, Y_test) 










