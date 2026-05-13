#!/usr/bin/env python3
"""
Model 4: NLP Classification - Prediction Script
Loads trained model and outputs predictions on raw test data.

Usage: python predict.py
Output: test_data/model4_results.csv
"""

import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# paths
MODEL_DIR = Path(__file__).resolve().parent / "saved_model"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model4_results.csv"

# same mapping used during training
EFFECTIVENESS_MAP = {
    'Highly Effective': 'Highly Effective',
    'Considerably Effective': 'Somewhat Effective',
    'Moderately Effective': 'Somewhat Effective',
    'Marginally Effective': 'Somewhat Effective',
    'Ineffective': 'Ineffective',
}


def clean_text(text):
    """same cleaning function from train.py - has to match exactly"""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'%u[0-9a-fA-F]{4}', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def find_test_file():
    """look for the medication feedback csv in test_data/"""
    # try a few likely names
    candidates = [
        "patient_medication_feedback.csv",
        "model4_test_data.csv",
        "test_medication_feedback.csv",
        "medication_feedback.csv",
    ]
    for name in candidates:
        p = TEST_DATA_DIR / name
        if p.exists():
            return p

    # fallback: grab any csv in test_data that has a benefitsReview column
    for p in TEST_DATA_DIR.glob("*.csv"):
        try:
            cols = pd.read_csv(p, nrows=0).columns.tolist()
            if 'benefitsReview' in cols:
                return p
        except Exception:
            continue

    raise FileNotFoundError(f"couldn't find NLP test data in {TEST_DATA_DIR}")


def main():
    # load model + vectorizer
    model = joblib.load(MODEL_DIR / "model.joblib")
    tfidf = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")

    # find and load raw test data
    test_path = find_test_file()
    print(f"loading test data from {test_path}")
    df = pd.read_csv(test_path)

    # figure out the id column - could be 'Patient ID' or something else
    id_col = None
    for col in ['Patient ID', 'patient_id', 'id', 'ID']:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        # just use the index as id
        df['id'] = range(len(df))
        id_col = 'id'

    # clean the review text (same preprocessing as training)
    df['clean_text'] = df['benefitsReview'].apply(clean_text)

    # handle any fully empty reviews - give them a blank string so tfidf doesn't break
    df['clean_text'] = df['clean_text'].fillna('')

    # vectorize and predict
    X_test = tfidf.transform(df['clean_text'])
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # confidence = max probability across classes for each sample
    confidence = np.max(probabilities, axis=1)

    # build output matching the template: id, predicted_class, confidence
    results = pd.DataFrame({
        'id': df[id_col].values,
        'predicted_class': predictions,
        'confidence': np.round(confidence, 4),
    })

    results.to_csv(OUTPUT_FILE, index=False)
    print(f"saved {len(results)} predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
