import os 
import numpy as np 
from PIL import Image 
from zipfile import ZipFile, BadZipFile 


IMG_SIZE = 200

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

path = os.path.join(base_dir, "data", "dataset.zip") 


def unzip_function(path, base_dir):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    extract_dir = os.path.join(base_dir, "data", "kaggle_data")
    caltech_dir = os.path.join(extract_dir, "caltech-101") 

    os.makedirs(extract_dir, exist_ok=True)

    # sprawdzenie, czy dane są już rozpakowane
    if os.path.exists(caltech_dir) and len(os.listdir(caltech_dir)) > 0:
        print(f"Folder {extract_dir} contains data")
        return caltech_dir

    try:
        with ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Unpacked files to {extract_dir}")
    except BadZipFile:
        raise Exception(f"The ZIP File is corrupted: {path}")

    return caltech_dir

extract_dir = unzip_function(path, base_dir)


X, Y = [], [] 

classes = [cls for cls in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, cls))]
print("Found classes:", classes)


for label, cls in enumerate(classes):
    cls_folder = os.path.join(extract_dir, cls)
    for img_file in os.listdir(cls_folder):
        img_path = os.path.join(cls_folder, img_file)

        if not os.path.isfile(img_path) or not img_file.lower().endswith(('.jpg', '.jpeg')):
            continue
        
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            X.append(np.array(img))
            Y.append(label)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue


X = np.array(X)
Y = np.array(Y)
