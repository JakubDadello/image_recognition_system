# Industrial-AI

## Project Description

This project builds a complete deep‑learning pipeline for automated steel surface defect classification, covering data extraction, preprocessing, and model training. It compares a custom ResNet architecture with a fine‑tuned ResNet50 to evaluate the effectiveness of domain‑specific learning versus transfer learning in industrial quality‑control scenarios.

## How to Run
Link to HuggingFace: https://huggingface.co/spaces/kejdixhug/steel_defect_detector
  
## Repository Structure
- `app` - production-ready deployment logic and API implementation (BentoML).
- `data/` – contains raw compressed data and fully extracted datasets (see `data/README.md` for details)
- `model` - serialized pre-trained Keras model artifacts.
- `notebooks/` – Jupyter notebooks for preprocessing (including data ingestion, data extraction and data loading), exploratory data analysis, and modeling experiments
- `src/` – Python scripts for data preprocessing, model definition, training, and utility functions
- `results/` – saved model weights, training logs, plots, and evaluation metrics
- `reports/` – visualizations, figures, and reports summarizing model performance and analysis insights


## Tech Stack

- Language: Python 3.11
- Libraries: Scikit-learn, Pandas, NumPy, Matplotlib, TensorFlow (Keras API)
- Tools: Canva (Presentation)
- Backend & Deployment: BentoML, Docker, Uvicorn


## Models Architecture & Strategy

This project evaluates and compares two distinct architectural strategies for industrial defect classification:

1. **Custom ResNet (Built from Scratch)** - Developed a tailored ResNet-18/34 inspired architecture implementing deep residual learning.
   - Designed a custom `ResidualBlock` class to handle identity mappings and alleviate the vanishing gradient problem.
   - Fully trained on the organized steel surface dataset to demonstrate the model's ability to learn domain-specific features without prior bias.

2. **Transfer Learning with ResNet50 (Fine-Tuning)** - Leveraged a pretrained ResNet50 backbone (ImageNet weights) as a high-level feature extractor.
   - Replaced the top layers with a custom global average pooling and a dense classification head tailored for the 6 specific steel defect categories.
   - Serves as a performance benchmark to evaluate how general-purpose features (ImageNet) adapt to specialized industrial textures.

This dual-model approach highlights the trade-off between **domain-specific training from scratch** and the **efficiency of transfer learning** in a specialized industrial context.

For the full CRISP-DM methodology, see [CRISP-DM.md](CRISP-DM.md)

