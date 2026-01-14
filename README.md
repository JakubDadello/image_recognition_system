# Industrial-AI

## Project Description

Image Recognition System is a convolutional neural network (CNN) project for classifying images from the Caltech-101 dataset (101 classes). The model uses custom residual blocks and transfer learning to improve feature extraction and convergence.

Data preprocessing includes resizing, normalization, and augmentation to enhance generalization. The network was trained for 10 epochs, monitoring loss and accuracy on both training and validation sets. Evaluation demonstrates effective classification across multiple classes, and insights from model performance guide potential improvements.

## How to Run

To run this project locally, you can use either Docker (recommended) or a standard Python environment.

### Using Docker 
* **Docker Compose:**
  ```bash
  docker-compose up --build

* **Build the image:**
  ```bash
  docker build -t industrial-ai .
* **Run the container:**
  ```bash
  docker run -p 8000:8000 industrial-ai

### Using Python (Local Environment)
* **Install requirements:**
  ```bash
  pip install -r requirements.txt
* **Launch FastAPI server:**
  ```bash
  uvicorn src.main:app --host 0.0.0.0 --port 8000
  
## Repository Structure
- `data/` – contains raw compressed data and fully extracted datasets (see `data/README.md` for details)
- `notebooks/` – Jupyter notebooks for preprocessing (including data ingestion, data extraction and data loading), exploratory data analysis, and modeling experiments
- `src/` – Python scripts for data preprocessing, model definition, training, and utility functions
- `results/` – saved model weights, training logs, plots, and evaluation metrics
- `reports/` – visualizations, figures, and reports summarizing model performance and analysis insights


## Tech Stack

- Language: Python 3.13.1
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, TensorFlow 
- Tools: Canva (Presentation)


## Models 

This project compares two approaches for image classification on the Caltech-101 dataset:

1. **Custom ResNet (from scratch)**  
   - Implemented a ResNet-34-like architecture using a custom ResidualBlock.  
   - Fully trained on our preprocessed dataset.  
   - Allows full control over network design and residual connections.

2. **Pretrained ResNet50 with custom classification head**  
   - Leveraged Keras' ResNet50 as a feature extractor (ImageNet weights).  
   - Added a custom Dense head for 101 classes.  
   - Used for benchmarking and comparison with the custom model.

This comparison highlights the trade-off between **training a network from scratch** versus **using transfer learning with a pretrained backbone**.


