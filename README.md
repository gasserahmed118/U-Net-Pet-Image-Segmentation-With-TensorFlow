U-Net Pet Segmentation with TensorFlow
Overview

This project implements a U-Net-based semantic segmentation model using TensorFlow 2.x to segment the Oxford-IIIT Pet Dataset.

The model leverages MobileNetV2 as the backbone and incorporates advanced techniques such as:

Dice + Crossentropy combined loss

Backbone fine-tuning

Data augmentation (flips, rotations, brightness/contrast, hue)

Mixed precision training (for faster training on GPU)

With these optimizations, the model achieves >95% pixel-wise accuracy on the test set.

Objectives

Train a U-Net model for pet image segmentation.

Apply data augmentation and backbone fine-tuning to improve performance.

Demonstrate high-accuracy segmentation (>95%) on the Oxford-IIIT Pet dataset.

Provide a Colab-ready implementation for easy reproducibility.

Project Structure
U-Net-Pet-Segmentation/
│
├─ notebook.ipynb           # Main Jupyter/Colab notebook
├─ README.md                # This file
├─ requirements.txt         # Required Python packages
├─ outputs/                 # Folder to save sample predictions
│
└─ utils.py                 # Optional: helper functions (mask creation, visualization)

Installation

Recommended: Use Google Colab for GPU support and to avoid version issues.

Clone the repository:

git clone <your-repo-url>
cd U-Net-Pet-Segmentation


Install dependencies:

pip install -r requirements.txt


If running locally, ensure NumPy < 2 to avoid TensorFlow import errors.

Requirements

Python 3.8+

TensorFlow 2.x

TensorFlow Datasets

Matplotlib

IPython (for clear_output)

TensorFlow Examples (for pix2pix upsampling)

Optional (for local GPU usage):

CUDA Toolkit

cuDNN

Usage
1. Run in Colab (Recommended)

Open notebook.ipynb in Google Colab.

Run the first cell to install packages:

!pip install tensorflow tensorflow-datasets
!pip install git+https://github.com/tensorflow/examples.git


Run all cells — the notebook automatically loads the dataset, trains the U-Net model, and displays predictions.

2. Key Functions

normalize(input_image, input_mask) → scales images and masks

load_image_train() → applies data augmentation for training

load_image_test() → prepares test images

unet_model() → builds the U-Net with MobileNetV2 backbone

dice_loss() → computes Dice loss

combined_loss() → Dice + SparseCategoricalCrossentropy

create_mask() → converts predictions to masks

show_predictions() → visualizes predicted masks

Training

Stage 1: Train top layers (backbone frozen)

Stage 2: Fine-tune backbone (low learning rate)

Stage 3 (Optional): Full fine-tuning with mixed precision

Callbacks used:

EarlyStopping → stops training when validation loss stops improving

ReduceLROnPlateau → reduces learning rate if loss plateaus

DisplayCallback → visualizes predictions after each epoch

Results

Training Accuracy: >95% after full fine-tuning

Validation Accuracy: ~95%

Sample Predictions:

Notes

Using Colab GPU is highly recommended; training on CPU will be slow.

Dataset version is 4.0.0 in tfds.load.

Ensure NumPy < 2 when running locally to avoid TensorFlow import errors.
