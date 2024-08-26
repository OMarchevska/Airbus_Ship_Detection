# Semantic Segmentation using Airbus Ship Detection Dataset

This repository is dedicated to solving the semantic segmentation problem using the [Airbus Ship Detection dataset](https://www.kaggle.com/competitions/airbus-ship-detection/data), which is part of the [Kaggle competition](https://www.kaggle.com/competitions/airbus-ship-detection).

## Instruments
- TensorFlow
- Keras
- KerasTuner
- pandas
- numpy

## Data
Total number of unique images: **192,556** (100%)
- Number of images with one or more targets to segment: **42,556** (22.0%)
- Number of images with no objects to segment: **150,000** (78.0%)

The high percentage of background images relative to images containing targets indicates an imbalance problem. This issue was further reinforced by pixel-level imbalance, where the objects to segment mostly occupy less than 1% of the total number of image pixels.

Image masks were provided in the form of run-length encoding on the pixel values, which were grouped by unique image ID and saved as per-image masks on the local disk to speed up the training process later.

## Dataset
Two sets of training and validation datasets were created:
- **Classification Pretraining**: Image/label pairs, where `1` indicates the image contains targets and `0` indicates background. The percentage of target images was 100%, and the background was 70% sampled.
- **Segmentation Pretraining & Fine-Tuning**: Image/mask pairs, where the resolution of data instances, depending on the pretraining/fine-tuning stage, was controlled using custom resizing logic embedded into the data pipeline. The percentage of target images was 100%, and the background was 10% sampled.

### Augmentation
A custom class was implemented to apply random transformations to both image and mask simultaneously, including horizontal/vertical flips, rotation, and brightness adjustments.

## Metric
The Dice similarity coefficient metric was used for semantic segmentation, measuring the similarity between predicted and ground truth masks. It is defined as twice the intersection of the predicted and ground truth masks divided by the sum of their sizes.

## Loss
A custom loss class was implemented, inspired by a paper comparing different losses for imbalanced data in semantic segmentation problems. The authors developed **Symmetric Unified Focal Loss** as a variant to tackle the imbalance problem more efficiently. This loss function combines Dice Loss and Binary Focal Loss to address class imbalance and improve pixel-wise segmentation accuracy. Dice Loss measures the overlap between predicted and ground truth masks, while Binary Focal Loss penalizes false positives and false negatives.

## Model
- **Base Model**: MobileNetV2
- **Segmentation Model**: The base model was converted into a U-Net-like architecture.
- **Total number of trainable parameters**: 6.5M

## Training
Each stage is supported by optimal learning rate finding and saving the best model locally.

1. **Pretraining**:
    - **Stage 1** (5 epochs): Base model (MobileNetV2) pretrained on the classification task (Ship/No Ship).
    - **Stage 2** (10 epochs): Base model converted into a segmentation model and trained further on the segmentation task.

2. **Fine-Tuning**:
    - **Stage 1** (8 epochs): Model fine-tuning using images of size 384x384.  
      - Training/validation dice scores: **0.8322** / **0.8460**
    - **Stage 2** (2 epochs): Model fine-tuning using images of size 512x512.  
      - Training/validation dice scores: **0.8221** / **0.8466**
    - **Stage 3** (2 epochs): Model fine-tuning using images of size 768x768.  
      - Training/validation dice scores: **0.8327** / **0.8503**

## Results
The combination of the custom loss function and the staged training process resulted in a dice score of **0.85** on images of the original size (768x768). The trained model was also evaluated on test images, showing good performance (low to zero false positives) even though most of the images provided were background.

