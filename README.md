# U-Net Pet Segmentation with TensorFlow

## Overview

This project implements a **U-Net-based semantic segmentation model** using TensorFlow 2.x to segment the **Oxford-IIIT Pet Dataset**.  

The model leverages **MobileNetV2 as the backbone** and incorporates advanced techniques such as:

- **Dice + Crossentropy combined loss**
- **Backbone fine-tuning**
- **Data augmentation** (flips, rotations, brightness/contrast, hue)
- **Mixed precision training** (for faster training on GPU)

With these optimizations, the model achieves **>95% pixel-wise accuracy** on the test set.

---

## Objectives

1. Train a U-Net model for pet image segmentation.
2. Apply data augmentation and backbone fine-tuning to improve performance.
3. Demonstrate high-accuracy segmentation (>95%) on the Oxford-IIIT Pet dataset.
4. Provide a **Colab-ready implementation** for easy reproducibility.

---

## Project Structure

U-Net-Pet-Segmentation/
│
├─ notebook.ipynb # Main Jupyter/Colab notebook
├─ README.md # This file
├─ requirements.txt # Required Python packages
├─ outputs/ # Folder to save sample predictions
│
└─ utils.py # Optional: helper functions (mask creation, visualization)



## Requirements
   1. Python 3.8+
   2. TensorFlow 2.x
   3. TensorFlow Datasets
   4. Matplotlib
   5. IPython (for clear_output)
   6. TensorFlow Examples (for pix2pix upsampling)
