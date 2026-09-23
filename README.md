# Multi-Class Disaster Image Classification

A deep learning computer vision project for classifying disaster images into four categories using a custom Convolutional Neural Network (CNN) and transfer learning with EfficientNetB0.

The project covers exploratory data analysis, image preprocessing, data augmentation, class imbalance handling, CNN development, transfer learning, fine-tuning, and model evaluation.

## Project Overview

The objective of this project is to automatically classify disaster images into four categories:

- Earthquake
- Landslide
- Urban Fire
- Water Disaster

The dataset contains **2,345 disaster images** collected across the four classes.

### Dataset Distribution

| Class | Number of Images |
|---|---:|
| Earthquake | 435 |
| Landslide | 456 |
| Urban Fire | 419 |
| Water Disaster | 1,035 |
| **Total** | **2,345** |

The dataset is imbalanced, with Water Disaster containing substantially more images than the other classes.

## Image Preprocessing

All images are resized to:

```text
224 × 224 pixels
```

The dataset is divided into training, validation, and test sets.

The training pipeline also applies data augmentation to improve model generalization.

For the custom CNN, augmentation includes:

- Horizontal flipping
- Rotation
- Zoom
- Contrast adjustment
- Translation

Validation and test images are not augmented.

## Handling Class Imbalance

Because the dataset distribution is imbalanced, class weighting is used during model training.

Instead of manually selecting the class weights, the project applies **Bayesian Optimization** to search for class weights that improve the macro F1-score.

This helps the model pay sufficient attention to minority classes without relying only on overall accuracy.

## Models

Two deep learning approaches were developed and compared.

### 1. Custom CNN

A Convolutional Neural Network was developed from scratch using TensorFlow and Keras.

The architecture uses convolutional layers for image feature extraction followed by pooling, dense layers, dropout, and a Softmax output layer for four-class classification.

The custom CNN achieved:

```text
Accuracy:  64.49%
Macro F1:  0.6527
```

The results showed that the CNN was able to learn disaster-related visual patterns, although its performance varied between classes.

### 2. EfficientNetB0 Transfer Learning

The second model uses **EfficientNetB0 pretrained on ImageNet** as the feature extraction backbone.

The original classification head is removed and replaced with custom layers containing:

- Global Average Pooling
- Batch Normalization
- Dense layers
- Dropout
- L2 regularization
- Softmax classification

Training is performed in two stages.

#### Stage 1 — Feature Extraction

The EfficientNetB0 backbone is initially frozen while the custom classification head is trained.

#### Stage 2 — Fine-Tuning

The final layers of the EfficientNetB0 backbone are unfrozen and trained using a smaller learning rate.

This allows the pretrained features to adapt more specifically to disaster imagery.

## Model Comparison

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Custom CNN | 64.49% | 0.6527 |
| **EfficientNetB0** | **82.45%** | **0.8098** |

EfficientNetB0 significantly improved classification performance compared with the custom CNN.

Accuracy increased by approximately **17.9 percentage points**, while macro F1 increased from **0.6527 to 0.8098**.

## EfficientNetB0 Final Results

The final EfficientNetB0 model achieved:

```text
Accuracy:  0.8245
Macro F1:  0.8098
```

### Classification Performance

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Earthquake | 0.7818 | 0.7963 | 0.7890 |
| Landslide | 0.6531 | 0.6667 | 0.6598 |
| Urban Fire | 0.8667 | 0.9286 | 0.8966 |
| Water Disaster | 0.9099 | 0.8783 | 0.8938 |

The strongest classification performance was achieved for **Urban Fire** and **Water Disaster**, while Landslide remained the most challenging class.

## Training Strategy

Several techniques were used to improve model generalization and training stability:

- Data augmentation
- Class weighting
- Bayesian Optimization
- Transfer learning
- Fine-tuning
- Dropout
- Batch normalization
- L2 regularization
- Early stopping
- Learning rate reduction
- Model checkpointing

## Project Workflow

```text
Disaster Image Dataset
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Image Resizing
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Data Augmentation
        │
        ▼
Class Weight Optimization
        │
        ├───────────────────────────┐
        ▼                           ▼
   Custom CNN               EfficientNetB0
        │                           │
        │                   Transfer Learning
        │                           │
        │                      Fine-Tuning
        │                           │
        └──────────────┬────────────┘
                       ▼
                 Model Evaluation
                       │
                       ▼
           Accuracy & Macro F1
```

## Project Structure

```text
disaster-image-classification/
│
├── PROJECT_DL_FINAL.ipynb
├── project-report.pdf
└── README.md
```

## File Description

`PROJECT_DL_FINAL.ipynb`

Contains the complete deep learning workflow including:

- Dataset exploration
- Image analysis
- Data preprocessing
- Data augmentation
- Class imbalance handling
- Bayesian class-weight optimization
- Custom CNN development
- EfficientNetB0 transfer learning
- Fine-tuning
- Model evaluation
- Model comparison

`project-report.pdf`

Contains the project report and documentation.

## Tech Stack

### Deep Learning

- TensorFlow
- Keras
- EfficientNetB0
- Convolutional Neural Networks
- Transfer Learning
- Fine-Tuning

### Machine Learning

- Scikit-learn
- Scikit-optimize
- Bayesian Optimization

### Data Processing & Visualization

- NumPy
- PIL
- Matplotlib

### Development

- Python
- Jupyter Notebook
- Google Colab

## Key Features

- Multi-class image classification
- Four disaster categories
- Custom CNN implementation
- EfficientNetB0 transfer learning
- Fine-tuning pretrained neural networks
- Data augmentation
- Class imbalance handling
- Bayesian Optimization for class weights
- Per-class performance evaluation
- CNN vs transfer-learning comparison

## Future Improvements

Possible future improvements include:

- Compare additional pretrained architectures such as ResNet and MobileNet
- Increase dataset size and class balance
- Apply more advanced image augmentation techniques
- Add Grad-CAM for model explainability
- Build an image-upload prediction interface
- Deploy the trained model as a web application
