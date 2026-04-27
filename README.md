# Automated Quality Control System

A deep learning-based system for automated visual inspection of products. This project uses transfer learning with ResNet50 to classify products as "Pass" or "Fail" based on synthetic image data.

## Overview

This system demonstrates an end-to-end quality control pipeline:
1. **Synthetic Data Generation** - Creates "Pass" (perfect circles) and "Fail" (defective) images
2. **Transfer Learning** - Uses ResNet50 pre-trained on ImageNet
3. **Model Training** - Trains a binary classifier for defect detection
4. **Evaluation** - Provides classification reports, confusion matrices, and visual results

## Requirements

```
pandas
numpy
opencv-python
tensorflow
scikit-learn
matplotlib
seaborn
```

Install dependencies:
```bash
pip install pandas numpy opencv-python tensorflow scikit-learn matplotlib seaborn
```

## Usage

Run the script:
```bash
python Automated_quality_control_system.py
```

The script will:
1. Generate 1,000 synthetic images (500 pass, 500 fail)
2. Split data into 70% training, 15% validation, 15% test
3. Train the model for 10 epochs
4. Evaluate and display results

## Model Details

- **Base Model**: ResNet50 (ImageNet weights)
- **Input Size**: 224x224x3
- **Custom Head**: GlobalAveragePooling2D → Dropout(0.5) → Dense(256) → Dense(1)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss**: Binary Crossentropy

## Output

The system outputs:
- Training history plot (loss/accuracy over epochs)
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Visual examples of predictions on test images

## GPU Note

For faster training, ensure a GPU is available. The script will warn if no GPU is detected.