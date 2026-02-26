# Image Classification (English vs Urdu)

A simple **image classification** project that detects whether a text image is **English** or **Urdu**.

It includes:
- Synthetic dataset generation (text rendered into images)
- Model training (lightweight CNN)
- Inference script to classify an image and route to the correct OCR pipeline

---

## Features

- Generate English/Urdu text images into dataset folders
- Train a CNN classifier on the generated dataset
- Save trained model + class index mapping
- Predict language of a given image with confidence score
- Route output to English OCR or Urdu OCR (hooks included)

---

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Pillow (PIL)

---
