import os
import json
import zipfile
import tempfile
import shutil

import cv2
import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras import models, layers

MODEL_PATH = os.path.join("model", "plant_seedling_model.keras")
CLASS_NAMES_PATH = os.path.join("model", "class_names.npy")

IMG_SIZE = 128
NUM_CLASSES = 12


def build_model():
    m = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (5, 5), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, (5, 5), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.5),
        layers.GlobalMaxPooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    return m


def load_model_weights(model_path):
    """Load weights from .keras file, bypassing config deserialization issues."""
    m = build_model()
    tmp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            zf.extractall(tmp_dir)
        weights_path = os.path.join(tmp_dir, "model.weights.h5")
        m.load_weights(weights_path)
    finally:
        shutil.rmtree(tmp_dir)
    return m


model = load_model_weights(MODEL_PATH)
CLASS_NAMES = np.load(CLASS_NAMES_PATH, allow_pickle=True)


def predict(image):
    """Preprocess a plant seedling image and return species predictions."""
    if image is None:
        return {}

    img = cv2.resize(image, (128, 128))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = img.astype(np.float32) / 255.0
    img = img.reshape(1, 128, 128, 3)

    predictions = model.predict(img, verbose=0)[0]
    return {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload a plant seedling image"),
    outputs=gr.Label(num_top_classes=5, label="Predicted Species"),
    title="Plant Seedlings Classifier",
    description=(
        "Upload an image of a plant seedling to identify its species. "
        "The CNN model classifies 12 species of plant seedlings from the "
        "[Plant Seedlings Dataset](https://www.kaggle.com/c/plant-seedlings-classification)."
    ),
    article=(
        "### Model Architecture\n"
        "4-block CNN with batch normalization and progressive dropout, "
        "trained on 128×128 images with data augmentation.\n\n"
        "### Species\n"
        "Black-grass, Charlock, Cleavers, Common Chickweed, Common wheat, "
        "Fat Hen, Loose Silky-bent, Maize, Scentless Mayweed, "
        "Shepherds Purse, Small-flowered Cranesbill, Sugar beet"
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
