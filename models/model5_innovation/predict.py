#!/usr/bin/env python3
"""
Model 5: Innovation — Prediction Script
=========================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model5_results.csv
"""
#to pull clean data and engineer features from data_pipeline
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import shared pipeline functions
from pipelines.data_pipeline import clean_data, engineer_features

# Paths
MODEL_PATH = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model"
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
OUTPUT_FILE = TEST_DATA_DIR / "model5_results.csv"


def load_model():
    #import model
    import joblib
    filepath = MODEL_PATH / "model.joblib"
    return joblib.load(filepath)


def predict(model, test_df):
   #use the test data

    df = test_df.copy()

    #clean the test data
    df = clean_data(df)
    df = engineer_features(df)

    #put test data encounter id in the results
    df_result = df[['encounter_id']].copy()
    df_result = df_result.rename(columns={"encounter_id": "id"}) 

    #drop all med columns
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone',
        'num_active_meds', 'n_meds_changed', 'n_meds_increased'
    ]

    existing_med_cols = [col for col in med_cols if col in df.columns]
    df = df.drop(columns=existing_med_cols, errors="ignore")

    #drop target and encounter id
    df = df.drop(columns=["diabetesMed", "encounter_id"], errors="ignore")


    #use model to make prediction and confidence score
    confidence_scores = model.predict_proba(df)[:, 1]
    predictions = model.predict(df)

    df_result["prediction"] = predictions
    df_result["confidence"] = confidence_scores
    df_result["metric_name"] = "f1_score"
    df_result["metric_value"] = 0.8770

    return df_result


def main():
    # Load model
    model = load_model()

    #set the test data
    test_df = pd.read_csv(TEST_DATA_DIR / "test_data_file.csv")

    #run the prediction
    results = predict(model, test_df)

    #sabve the results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
