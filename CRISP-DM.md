# Business Understanding

Goal: Automated Quality Control in Steel Manufacturing.

In traditional steel manufacturing, surface defect inspection is often performed manually by human operators. This process is prone to errors due to fatigue, subjective judgment, and the high speed of production lines.
The objective of this project is to develop a Deep Learning solution based on the ResNet architecture to automatically classify surface defects in hot-rolled steel strips.

### Key Objectives:

Accuracy: Achieve high classification precision to minimize "false passes" of defective material.

Efficiency: Reduce the workload on human inspectors by filtering out clear cases of defects.

Scalability: Create a model that can be deployed on edge devices near the production line for real-time inference.

# Data Understanding
Dataset: NEU Surface Defect Database.

The dataset contains 1,800 grayscale images categorized into six distinct types of surface defects. Each class represents a specific metallurgical or production-related issue.

### Classes of Interest:

1. Crazing: Interconnected cracks on the surface.

2. Inclusion: Non-metallic particles trapped in the steel.

3. Patches: Localized surface irregularities.

3. Pitted Surface: Small cavities caused by corrosion or mechanical damage.

4. Rolled-in Scale: Iron oxides pressed into the surface during rolling.

5. Scratches: Linear abrasions from handling or machinery.

### Data Characteristics:

1. Original Image Size: 200x200 pixels.

2. Format: RGB (processed as 3-channel input for ResNet compatibility).

3. Split Strategy: The dataset follows a predefined split to ensure benchmark consistency:

- Training: 85% (2352 images)

- Validation: 11% (295 images)

- Testing: 4% (113 images)

# Data Preparation
To ensure high model performance and robustness, a custom automated pipeline was developed.

### Wrangling & Organization:

1. Flat File Parsing: Since the source files were provided in a flat structure with labels in filenames (e.g., inclusion_1.jpg), an automated script was implemented to reorganize them into a directory-based structure (/train/inclusion/).

2. Split Preservation: The pipeline strictly maintains the original split (85/11/4) to allow for fair evaluation against industry standards.

### Preprocessing & Augmentation:

1. Normalization: Pixel values are rescaled from [0, 255] to the [0, 1] range to facilitate faster convergence of the ResNet model.

2. On-the-fly Augmentation: To prevent overfitting, the training pipeline applies random horizontal/vertical flips and rotations. This increases the model's ability to generalize to defects appearing at different angles.

3. Performance Optimization: Leveraging the tf.data API, the pipeline uses Prefetching and Autotuning. This ensures the CPU prepares the next batch of images while the GPU is busy training, significantly reducing training time.

# Modeling
In this project, we evaluate two different approaches to solving the surface defect classification problem: a high-level Transfer Learning approach and a low-level Custom Residual implementation.

###  Model 1: ResNet-50 (Transfer Learning)

Architecture: Standard ResNet-50 backbone with a custom-added classification head.

Implementation: Developed using tf.keras.applications.ResNet50.

Strategy: Fine-tuning the top layers while keeping the ImageNet pre-trained weights in the base frozen to act as a universal feature extractor.

### Model 2: Custom ResNet (Built from Scratch) 
This model represents a deep-dive into the architecture's mechanics. Instead of using a pre-packaged model, we implemented the Residual Learning logic manually.

The architecture is built upon a custom ResidualBlock component. The implementation focuses on the mathematical foundation of skip connections:

1. The Main Path $F(x)$: Consists of a series of $3 \times 3$ and $1 \times 1$ Convolutional layers, Batch Normalization, and ReLU activations.

2. The Skip Connection (Identity): A parallel path that carries the original input $x$.
   
3. Projection Shortcut: A crucial technical detail in our script. When spatial dimensions decrease (stride $> 1$), we apply a $1 \times 1$ convolution to the shortcut path to ensure the tensors match for the final addition.
   
4. Integration: The final output is calculated as $y = \text{ReLU}(F(x) + \text{Shortcut}(x))$.









