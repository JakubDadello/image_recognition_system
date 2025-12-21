# Project Description

Image Recognition System is a convolutional neural network (CNN) project for classifying images from the Caltech-101 dataset (101 classes). The model uses custom residual blocks and transfer learning to improve feature extraction and convergence.

Data preprocessing includes resizing, normalization, and augmentation to enhance generalization. The network was trained for 5 epochs, monitoring loss and accuracy on both training and validation sets. Evaluation demonstrates effective classification across multiple classes, and insights from model performance guide potential improvements.

Technologies: Python, TensorFlow/Keras, NumPy, Matplotlib, Pandas

## Repository Structure
- `data/` – contains raw compressed data and fully extracted datasets (see `data/README.md` for details)
- `notebooks/` – Jupyter notebooks for preprocessing (including data ingestion and data extraction), exploratory data analysis, and modeling experiments
- `src/` – Python scripts for data preprocessing, model definition, training, and utility functions
- `results/` – saved model weights, training logs, plots, and evaluation metrics
- `reports/` – visualizations, figures, and reports summarizing model performance and analysis insights

