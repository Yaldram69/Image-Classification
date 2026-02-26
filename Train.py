# ========================
# TRAINING CODE
# ========================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

# ========================
# CONFIG
# ========================
IMG_HEIGHT = 100
IMG_WIDTH = 300
BATCH_SIZE = 32
EPOCHS = 20
DATASET_DIR = "dataset"  # Subfolders: 'english', 'urdu'

# ========================
# DATA AUGMENTATION & LOADER
# ========================
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

# Save class indices
print("Class indices:", train_data.class_indices)
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump(train_data.class_indices, f, indent=4, ensure_ascii=False)

# ========================
# LIGHTWEIGHT CNN MODEL
# ========================
model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output: 2 classes (english, urdu)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========================
# TRAINING
# ========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ========================
# SAVE
# ========================
model.save("lightweight_language_classifier.h5")
