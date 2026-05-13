"""
Model 1: Readmission Risk — Traditional ML — Predict
=====================================================
Loads the saved model and runs predictions on raw test data.
Handles all preprocessing internally — test data comes in messy,
predictions go out clean.

Usage (from project root):
    python models/model1_traditional_ml/predict.py

Input:
    test_data/  (any CSV with the same columns as patient_encounters_2023.csv)

Output:
    test_data/model1_results.csv  (columns: id, prediction, probability, confidence)
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# this file is at models/model1_traditional_ml/predict.py
# so project root is 2 levels up
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    clean_data,
    engineer_features,
    find_test_csv,
)

SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'
TEST_DIR = PROJECT_ROOT / 'test_data'
OUTPUT_PATH = TEST_DIR / 'model1_results.csv'

TARGET = 'readmission_binary'


def load_artifacts():
    # load the saved model and the feature column list from training
    model_path = SAVED_MODEL_DIR / 'model.joblib'
    cols_path = SAVED_MODEL_DIR / 'feature_cols.joblib'

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {model_path}\n"
            f"Run train.py first to generate the model."
        )
    if not cols_path.exists():
        raise FileNotFoundError(
            f"No feature cols found at {cols_path}\n"
            f"Run train.py first to generate feature_cols.joblib."
        )

    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    print(f'model loaded      → {model_path}')
    print(f'feature cols loaded → {len(feature_cols)} columns')
    return model, feature_cols


def load_test_data():
    # find the test CSV in test_data/ — uses find_test_csv from the pipeline
    # which filters out results files and looks for encounter-shaped data
    csv_path = find_test_csv(
        TEST_DIR,
        expected_columns=['encounter_id', 'readmitted'],
        name_hint='encounter'
    )
    print(f'test data found   → {csv_path.name}')
    df = pd.read_csv(csv_path)
    print(f'test shape        → {df.shape}')
    return df


def prepare_test_data(df, feature_cols):
    # run the same cleaning and feature engineering as train.py
    df = clean_data(df)
    df = engineer_features(df)

    # grab ids after cleaning so the length matches X exactly
    ids = df['encounter_id'].copy() if 'encounter_id' in df.columns else pd.Series(range(len(df)))

    # drop the target if it somehow exists in test data
    df = df.drop(columns=[TARGET, 'readmitted'], errors='ignore')
    df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')

    # one-hot encode any remaining object columns — same as train.py
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        print(f'  one-hot encoding: {obj_cols}')
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # force numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # align to training columns exactly
    df = df.reindex(columns=feature_cols, fill_value=0)

    print(f'  features after alignment: {df.shape[1]}')
    return df, ids


def predict(model, X, ids):
    # run predictions and build the output dataframe
    proba = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    # confidence = how far the probability is from 0.5
    # 1.0 = completely certain, 0.0 = totally uncertain
    confidence = np.abs(proba - 0.5) * 2

    results = pd.DataFrame({
        'id': ids.values,
        'prediction': preds.astype(int),
        'probability': np.round(proba, 4),
        'confidence': np.round(confidence, 4),
    })

    return results


def main():
    print('=' * 50)
    print('  Model 1: Readmission Risk — Predict')
    print('=' * 50)

    # 1. load saved model and feature columns
    model, feature_cols = load_artifacts()

    # 2. load raw test data
    df = load_test_data()

    # 3. run through the same preprocessing pipeline as training
    print('\npreparing test data...')
    X, ids = prepare_test_data(df, feature_cols)

    # 4. generate predictions
    print('\nrunning predictions...')
    results = predict(model, X, ids)

    # 5. save output to test_data/model1_results.csv
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f'\nresults saved → {OUTPUT_PATH}')
    print(f'total predictions : {len(results)}')
    print(f'predicted readmit : {results["prediction"].sum()} ({results["prediction"].mean()*100:.1f}%)')
    print('\n✅ model 1 prediction complete.')


if __name__ == '__main__':
    main()
