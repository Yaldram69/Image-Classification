import sys
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ========================
# CONFIG
# ========================
IMG_SIZE    = 128
MODEL_PATH  = "best_language_classifier_tl.keras"
CLASS_MAP   = "class_indices.json"

# ========================
# LOAD MODEL & CLASS MAP
# ========================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_MAP, 'r', encoding='utf-8') as f:
    class_indices = json.load(f)

# invert mapping: {0: 'english', 1: 'urdu'}
idx_to_label = {v: k for k, v in class_indices.items()}

# ========================
# PREDICT FUNCTION
# ========================
def predict(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # shape (1,128,128,3)

    preds = model.predict(arr)[0]
    idx   = int(np.argmax(preds))
    label = idx_to_label[idx]
    conf  = float(preds[idx])
    return label, conf

# ========================
# MAIN
# ========================
if __name__ == "__main__":

    img_path = r'C:\Users\works\Downloads\Screenshot 2025-05-18 105740.png'

    label, confidence = predict(img_path)
    print(f"Image : {img_path}")
    print(f"Prediction: {label.upper()} (confidence: {confidence:.3f})")

    if label == "urdu":
        print("⮕ Routing to URDU OCR")
        # urdu_ocr(img_path)
    else:
        print("⮕ Routing to ENGLISH OCR")
        # english_ocr(img_path)
