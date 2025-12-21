import os
import numpy as np
from PIL import Image
from zipfile import ZipFile, BadZipFile

# =========================
# CONFIGURATION / CONSTANTS
# =========================
IMG_SIZE = 200  # Target size for images (200x200) to standardize input for CNN

# Set the base project directory (two levels up from this file)
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the compressed dataset
path_rd = os.path.join(base_path, "data", "dataset.zip")


# =========================
# DATA INGESTION / EXTRACTION
# =========================
def unzip_function(path, base):
    """
    Unzip the dataset ZIP file into 'data/kaggle_dataset'.
    
    Args:
        path (str): Path to the ZIP file containing the dataset.
        base (str): Base directory of the project (BASE_DIR).

    Returns:
        str: Path to the extracted dataset directory.
    
    Raises:
        FileNotFoundError: If the ZIP file does not exist.
        Exception: If the ZIP file is corrupted.
    """
    # Check if the ZIP file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Destination directory for extracted data
    extract_dir = os.path.join(base, "data", "kaggle_dataset")
    caltech_dir = os.path.join(extract_dir, "caltech-101")

    # Create the directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)

    # If data already exists, skip extraction
    if os.path.exists(caltech_dir) and len(os.listdir(caltech_dir)) > 0:
        print(f"Folder {caltech_dir} already contains data")
        return extract_dir

    try:
        # Extract ZIP file
        with ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Unpacked files to {extract_dir}")

    except BadZipFile:
        # Handle corrupted ZIP file
        raise Exception(f"The ZIP file is corrupted: {path}")

    return extract_dir


# Extract the dataset
extract_dir = unzip_function(path_rd, base_path)


# =========================
# DATA LOADING
# =========================
# Lists to store images and labels
X, Y = [], []

# Detect classes (folders) in the dataset
classes = [
    cls for cls in os.listdir(extract_dir)
    if os.path.isdir(os.path.join(extract_dir, cls))
]

print(f"Found classes: {classes}")


# =========================
# IMAGE PREPROCESSING
# =========================
for label, cls in enumerate(classes):
    cls_folder = os.path.join(extract_dir, cls)
    for img_file in os.listdir(cls_folder):
        img_path = os.path.join(cls_folder, img_file)

        # Skip files that are not JPEG images
        if not os.path.isfile(img_path) or not img_file.lower().endswith(('.jpg', '.jpeg')):
            continue

        try:
            # Open image and convert to RGB
            img = Image.open(img_path).convert("RGB")

            # Resize image to fixed dimensions (IMG_SIZE x IMG_SIZE)
            img = img.resize((IMG_SIZE, IMG_SIZE))

            # Convert image to numpy array and append to dataset
            X.append(np.array(img))
            Y.append(label)
        except Exception as e:
            # Handle errors when loading individual images
            print(f"Failed to load {img_path}: {e}")
            continue

# Convert lists to numpy arrays ready for model input
X = np.array(X)
Y = np.array(Y)