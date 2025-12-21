import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def data_preprocessing(
    X,
    Y,
    IMG_SIZE=224,
    test_size=0.20,
    val_size=0.25,
    batch_size=32,
    num_classes=101
):
    """
    Perform full data preprocessing pipeline:
    - train/validation/test split
    - image resizing and normalization
    - label encoding
    - data augmentation

    Args:
        X (np.ndarray): Array of input images.
        Y (np.ndarray): Array of class labels.
        IMG_SIZE (int): Target image size (IMG_SIZE x IMG_SIZE).
        test_size (float): Proportion of test data.
        val_size (float): Proportion of validation data (from remaining data).
        batch_size (int): Batch size for training generator.
        num_classes (int): Number of target classes.

    Returns:
        train_generator: Keras data generator with augmentation for training.
        (X_val, Y_val): Validation dataset.
        (X_test, Y_test): Test dataset.
    """

    # =========================
    # DATA SPLITTING
    # =========================
    # Split data into train+val and test sets (stratified to preserve class distribution)
    X_rest, X_test, Y_rest, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=42,
        stratify=Y
    )

    # Split remaining data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_rest, Y_rest,
        test_size=val_size,
        random_state=42,
        stratify=Y_rest
    )

    # =========================
    # IMAGE RESIZING
    # =========================
    # Resize all images to a fixed resolution expected by CNN architectures
    X_train = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy() for img in X_train])
    X_val   = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy() for img in X_val])
    X_test  = np.array([tf.image.resize(img, (IMG_SIZE, IMG_SIZE)).numpy() for img in X_test])

    # =========================
    # NORMALIZATION
    # =========================
    # Normalize pixel values to range [0, 1] for numerical stability
    X_train = X_train.astype("float32") / 255.0
    X_val   = X_val.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0

    # =========================
    # LABEL ENCODING
    # =========================
    # Convert labels from 1-based indexing to 0-based indexing
    Y_train = np.array(Y_train) - 1
    Y_val   = np.array(Y_val) - 1
    Y_test  = np.array(Y_test) - 1

    # One-hot encode labels for multi-class classification
    Y_train = to_categorical(Y_train, num_classes=num_classes)
    Y_val   = to_categorical(Y_val, num_classes=num_classes)
    Y_test  = to_categorical(Y_test, num_classes=num_classes)

    # =========================
    # DATA AUGMENTATION
    # =========================
    # Apply real-time data augmentation to improve generalization
    data_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # Fit generator on training data (required for some transformations)
    data_gen.fit(X_train)

    # Create training data generator
    train_generator = data_gen.flow(
        X_train,
        Y_train,
        batch_size=batch_size
    )

    return train_generator, (X_val, Y_val), (X_test, Y_test)