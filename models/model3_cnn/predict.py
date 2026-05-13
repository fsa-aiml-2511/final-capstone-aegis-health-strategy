#!/usr/bin/env python3
"""
Model 3: CNN — Prediction Script
==================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model3_results.csv
"""
import pandas as pd
from pathlib import Path
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Updated file paths to avoid issues with running this code
# I found that the previous code would error out if the terminal was not in a particular folder
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

MODEL_PATH = BASE_DIR / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
IMAGE_DIR = TEST_DATA_DIR / "images"
OUTPUT_FILE = TEST_DATA_DIR / "model3_results.csv"

def load_model():
    # Load saved model and return for predictions
    model = tf.keras.models.load_model(MODEL_PATH / "cnn_model.keras")
    return model

def load_and_preprocess_images(image_dir):

    images, ids = [], []
    # Load and preprocess images to predict
    for img_path in sorted(Path(image_dir).glob("*.png")):
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        ids.append(img_path.name)
    return np.array(images), ids

def predict(model, images, image_ids):
    # Feed all assessment images through the model
    raw_predictions = model.predict(images).flatten()

    # Generate binary predictions and confidence scores
    predicted_classes = (raw_predictions >= 0.5).astype(int)
    confidence_scores = raw_predictions
    
    # Build DataFrame with image_id, predicted_class, and confidence
    results = pd.DataFrame({
        "image_id": image_ids,
        "predicted_class": predicted_classes,
        "confidence": confidence_scores,
    })

    return results

def main():
    # Load model
    model = load_model()

    # Load test images from test_data/ image folder
    images, image_ids = load_and_preprocess_images(TEST_DATA_DIR / "images")

    # Get prediction results
    results = predict(model, images, image_ids)

    # Write results to test_data folder and print success message
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
