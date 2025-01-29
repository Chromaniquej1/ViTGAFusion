# ViTGAFusion

# GAF-NAU: Gramian Angular Field Encoded Neighborhood Attention Vision Transformer for Hyperspectral Image Classification

This repository contains the implementation of a novel approach for pixel-wise hyperspectral image classification using a **learnable Gramian Angular Field (GAF)** module fused with a **Vision Transformer (ViT)**. The project explores the integration of GAF encoding with ViT to improve feature representation and classification accuracy in hyperspectral imaging tasks.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Key Features](#key-features)
3. [Architecture Overview](#architecture-overview)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Dataset](#dataset)
7. [Model Architecture](#model-architecture)
8. [Training and Evaluation](#training-and-evaluation)
9. [Results](#results)
10. [Visualization](#visualization)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)
14. [Contact](#contact)

---

## Introduction

Hyperspectral images (HSI) contain rich spectral information, making them valuable for applications like remote sensing, agriculture, and environmental monitoring. However, classifying HSI data is challenging due to high dimensionality and limited labeled samples. This project introduces a **learnable GAF module** to encode spectral signatures into spatial representations, which are then processed by a **Vision Transformer (ViT)** for pixel-wise classification.

---

## Key Features

- **Learnable GAF Module**: A multi-channel GAF encoder that transforms spectral signatures into spatial representations.
- **Vision Transformer (ViT)**: A state-of-the-art transformer-based architecture for image classification.
- **Pixel-Wise Classification**: Designed for hyperspectral image classification at the pixel level.
- **Attention Mechanism**: Incorporates channel attention to weigh the importance of different GAF channels.
- **End-to-End Training**: The entire model is trained end-to-end using PyTorch and Hugging Face's `transformers` library.

---

## Architecture Overview

The architecture consists of three main components:

1. **Multi-Channel Learnable GAF Module**:
   - Encodes spectral signatures into spatial representations using learnable linear transformations.
   - Applies channel attention to weigh the importance of each GAF channel.

2. **Vision Transformer (ViT)**:
   - Processes the GAF-encoded images using a transformer-based architecture.
   - Outputs class probabilities for each pixel.

3. **Training Pipeline**:
   - Uses the Indian Pines dataset for training and evaluation.
   - Implements cross-entropy loss with class weighting to handle imbalanced data.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/GAF-NAU-ViT.git
   cd GAF-NAU-ViT

## Model Architecture

### 1. Multi-Channel Learnable GAF Module

- **Input**: Spectral signatures of shape `(batch_size, num_spectral_bands)`.
- **GAF Encoders**:
  - A set of learnable linear layers transforms each spectral signature into a spatial representation (GAF image).
  - Each GAF image has a size of `(image_size, image_size)`.
- **Channel Attention**:
  - A 1x1 convolutional layer followed by a softmax activation is used to compute attention weights for each GAF channel.
- **Output**: A weighted combination of GAF images, shaped `(batch_size, num_channels, image_size, image_size)`.

### 2. Vision Transformer (ViT)

- **Input**: Resized GAF images of shape `(batch_size, num_channels, 224, 224)`.
- **ViT Configuration**:
  - Image size: `224x224`
  - Patch size: `16x16`
  - Number of channels: `32`
  - Hidden size: `384`
  - Number of attention heads: `4`
  - Number of layers: `4`
- **Output**: Class probabilities for each pixel.

### 3. Loss Function

- **Cross-Entropy Loss**:
  - Weighted by class frequencies to handle imbalanced data.   

## Dataset

The **Indian Pines dataset** is used for training and evaluation. It consists of:

- **Hyperspectral data**: `Indian_pines_corrected.mat`
- **Ground truth labels**: `Indian_pines_gt.mat`

### Preprocessing Steps:

1. **Data Reshaping**:
   - The hyperspectral data is reshaped from `(145, 145, 200)` to `(21025, 200)` (flattened spatial dimensions).

2. **Masking Unlabeled Pixels**:
   - Pixels with label `0` (unlabeled) are removed.

3. **Normalization**:
   - Spectral signatures are normalized to have zero mean and unit variance.

4. **Train-Validation-Test Split**:
   - The dataset is split into training (60%), validation (20%), and test (20%) sets using stratified sampling.
  

