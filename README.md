# Image Classification Using Pre-Trained Vision Transformer

A PyTorch implementation for fine-tuning Vision Transformer (ViT-B-16) models on custom image classification datasets.

## Overview

This project leverages transfer learning with pre-trained Vision Transformer models to build efficient image classifiers. The modular architecture enables easy customization for various classification tasks.

## Features

- **Pre-trained ViT-B-16** - Leverages ImageNet pre-trained weights
- **Data Splitting** - Automated train/validation dataset partitioning
- **Transfer Learning** - Fine-tune backbone with frozen base parameters
- **PyTorch Integration** - GPU acceleration support via CUDA
- **Helper Functions** - Reusable utilities for training and evaluation

## Project Structure

```
├── code.ipynb              # Main training notebook
├── helper_functions.py     # Utility functions (plotting, accuracy metrics)
├── going_modular/          # Modular code components
└── README.md              # This file
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

## Usage

### Dataset Preparation

Organize your dataset with class subdirectories:
```
dataset/
├── class_1/
├── class_2/
└── class_3/
```

Split data using helper functions:
```python
from helper_functions import split_images
split_images("path/to/source", "path/to/dest", split_ratio=0.8)
```

### Training

Execute the Jupyter notebook:
```bash
jupyter notebook code.ipynb
```

## Model Details

- **Architecture:** Vision Transformer Base (ViT-B-16)
- **Backbone:** Frozen pre-trained parameters (transfer learning)
- **Device:** Auto-detects GPU/CPU availability
- **Customizable:** Easily adapt classifier head for different output classes
