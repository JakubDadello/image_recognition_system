import os
import shutil
import logging
import tensorflow as tf
from zipfile import ZipFile, BadZipFile


# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPipeline:
    def __init__(self, path_rd, extract_to, img_size=(200, 200), batch_size=32):
        """
        Initialize the data pipeline.

        Args:
            path_rd (str): Path to the compressed (ZIP) dataset file.
            extract_to (str): Destination directory for extracted dataset.
            img_size (tuple): Target image dimensions (height, width).
            batch_size (int): Size of training batches.
        """
        self.path_rd = path_rd
        self.extract_to = extract_to
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = [
            "crazing", 
            "inclusion", 
            "patches", 
            "pitted_surface", 
            "rolled-in_scale", 
            "scratches"
        ]

    # ---------------------------------------------------------
    # Main pipeline engine
    # ---------------------------------------------------------
    def running_engine(self):
        logging.info("Starting data pipeline...")
        self.extract_data()
        self.organize_flat_structure()
        return self.create_datasets()

    # ---------------------------------------------------------
    # Step 1: Extract ZIP
    # ---------------------------------------------------------
    def extract_data(self):
        if not os.path.exists(self.extract_to) or len(os.listdir(self.extract_to)) == 0:
            logging.info(f"Extracting dataset from {self.path_rd}...")
            try:
                with ZipFile(self.path_rd, "r") as zip_file:
                    zip_file.extractall(self.extract_to)
                logging.info("Extraction completed successfully.")
            except BadZipFile:
                logging.error("Dataset ZIP is corrupted.")
                raise RuntimeError("Dataset ZIP is corrupted.")
        else:
            logging.info("Dataset already extracted. Skipping extraction.")

    # ---------------------------------------------------------
    # Step 2: Organize dataset into class folders
    # ---------------------------------------------------------
    def organize_flat_structure(self):
        logging.info("Organizing images into class-specific folders...")

        train_path = os.path.join(self.extract_to, "train")

        if not os.path.exists(train_path):
            raise RuntimeError("Training folder missing in dataset.")
        
        for filename in os.listdir(train_path):
            file_path = os.path.join(train_path, filename)

            if os.path.isdir(file_path):
                continue

            for class_name in self.class_names:
                if filename.lower().startswith(class_name.lower()):
                    target_dir = os.path.join(train_path, class_name)
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.move(file_path, os.path.join(target_dir, filename))
                    break

        logging.info("Organization step completed.")

    # ---------------------------------------------------------
    # Step 3: Create tf.data datasets
    # ---------------------------------------------------------
    def create_datasets(self):
        logging.info("Creating TensorFlow datasets...")

        train_path = os.path.join(self.extract_to, "train")

        full_dataset = tf.keras.utils.image_dataset_from_directory(
            train_path,
            batch_size = self.batch_size,
            image_size = self.img_size,
            label_mode = "categorical",
            shuffle=True,
            seed=123,
        )

        # --- Compute sizes --- 
        dataset_size = full_dataset.cardinality().numpy() 
        val_size = dataset_size // 10 
        test_size = dataset_size // 10 
        train_size = dataset_size - val_size - test_size 

        # --- Split dataset ---
        train_data = full_dataset.take(train_size)
        remain_data = full_dataset.skip(train_size)
        val_data = remain_data.take(val_size)
        test_data = remain_data.skip(val_size)

        # Augmentation 
        augmentation = tf.keras.Sequential([ 
            tf.keras.layers.RandomFlip("horizontal_and_vertical"), 
            tf.keras.layers.RandomRotation(0.2) ]) 
        
        train_data = train_data.map(
            lambda x, y: (augmentation(x, training=True), y), 
            num_parallel_calls=tf.data.AUTOTUNE ) 

        # Cache + prefetch 
        train_data = train_data.prefetch(tf.data.AUTOTUNE) 
        val_data = val_data.cache().prefetch(tf.data.AUTOTUNE) 
        test_data = test_data.cache().prefetch(tf.data.AUTOTUNE) 
        
        logging.info("Datasets created successfully.") 
        
        return train_data, val_data, test_data

# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    pipeline = DataPipeline("data/dataset.zip", "data/steel_data")
    train_data, val_data, test_data = pipeline.running_engine()
    logging.info("Pipeline completed successfully.")