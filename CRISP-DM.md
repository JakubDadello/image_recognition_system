# Business Understanding

Goal: Automated Quality Control in Steel Manufacturing.

In traditional steel manufacturing, surface defect inspection is often performed manually by human operators. This process is prone to errors due to fatigue, subjective judgment, and the high speed of production lines.
The objective of this project is to develop a Deep Learning solution based on the ResNet architecture to automatically classify surface defects in hot-rolled steel strips.

## Key Objectives:

Accuracy: Achieve high classification precision to minimize "false passes" of defective material.

Efficiency: Reduce the workload on human inspectors by filtering out clear cases of defects.

Scalability: Create a model that can be deployed on edge devices near the production line for real-time inference.

### Data Understanding
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

