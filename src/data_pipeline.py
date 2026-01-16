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

        for split in ["train", "valid", "test"]:
            split_path = os.path.join(self.extract_to, split)

            if not os.path.exists(split_path):
                logging.warning(f"Split folder missing: {split}")
                continue

            for filename in os.listdir(split_path):
                file_path = os.path.join(split_path, filename)

                # Skip directories
                if os.path.isdir(file_path):
                    continue

                # Move file into correct class folder
                for class_name in self.class_names:
                    if filename.lower().startswith(class_name.lower()):
                        target_dir = os.path.join(split_path, class_name)
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
        val_path = os.path.join(self.extract_to, "valid")
        test_path = os.path.join(self.extract_to, "test")

        # Minimal validation
        if not os.path.exists(train_path):
            raise RuntimeError("Training folder is missing. Dataset not prepared correctly.")

        # Load datasets
        train_data = tf.keras.utils.image_dataset_from_directory(
            train_path, image_size=self.img_size, batch_size=self.batch_size,
            label_mode="categorical"
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            val_path, image_size=self.img_size, batch_size=self.batch_size,
            label_mode="categorical"
        )

        test_data = tf.keras.utils.image_dataset_from_directory(
            test_path, image_size=self.img_size, batch_size=self.batch_size,
            label_mode="categorical"
        )

        # Preprocessing layers
        rescaler = tf.keras.layers.Rescaling(1.0 / 255)
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2)
        ])

        # Shuffle + augmentation for training
        train_data = train_data.shuffle(1000)
        train_data = train_data.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Rescaling for all datasets
        train_data = train_data.map(
            lambda x, y: (rescaler(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        val_data = val_data.map(
            lambda x, y: (rescaler(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        test_data = test_data.map(
            lambda x, y: (rescaler(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # Cache validation & test for performance
        val_data = val_data.cache()
        test_data = test_data.cache()

        # Prefetch for performance
        return (
            train_data.prefetch(tf.data.AUTOTUNE),
            val_data.prefetch(tf.data.AUTOTUNE),
            test_data.prefetch(tf.data.AUTOTUNE)
        )


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    pipeline = DataPipeline("data/dataset.zip", "data/steel_data")
    train_data, val_data, test_data = pipeline.running_engine()
    logging.info("Pipeline completed successfully.")