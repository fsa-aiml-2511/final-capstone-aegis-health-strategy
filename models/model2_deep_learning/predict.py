"""
Model 2: Readmission Risk — Deep Neural Network — Predict
==========================================================
Loads the saved DNN model and runs predictions on raw test data.
Handles all preprocessing internally — same pipeline as train.py.

Usage (from project root):
    python models/model2_deep_learning/predict.py

Input:
    test_data/  (any CSV with the same columns as patient_encounters_2023.csv)

Output:
    test_data/model2_results.csv  (columns: id, prediction, probability, confidence)
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf

warnings.filterwarnings('ignore')

# this file is at models/model2_deep_learning/predict.py
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
OUTPUT_PATH = TEST_DIR / 'model2_results.csv'

TARGET = 'readmission_binary'


def load_artifacts():
    # load the saved keras model plus the three preprocessing artifacts
    # all four must exist — if train.py ran successfully they will be here
    model_path = SAVED_MODEL_DIR / 'model.keras'
    cols_path = SAVED_MODEL_DIR / 'feature_cols.joblib'
    imputer_path = SAVED_MODEL_DIR / 'imputer.joblib'
    scaler_path = SAVED_MODEL_DIR / 'scaler.joblib'

    for p in [model_path, cols_path, imputer_path, scaler_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing artifact: {p}\n"
                f"Run train.py first to generate all saved model files."
            )

    model = tf.keras.models.load_model(model_path)
    feature_cols = joblib.load(cols_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)

    print(f'model loaded       → {model_path}')
    print(f'feature cols loaded → {len(feature_cols)} columns')
    return model, feature_cols, imputer, scaler


def load_test_data():
    # find the test CSV in test_data/ — skips results files automatically
    csv_path = find_test_csv(
        TEST_DIR,
        expected_columns=['encounter_id', 'readmitted'],
        name_hint='encounter'
    )
    print(f'test data found    → {csv_path.name}')
    df = pd.read_csv(csv_path)
    print(f'test shape         → {df.shape}')
    return df


def prepare_test_data(df, feature_cols, imputer, scaler):
    # run the exact same preprocessing as train.py
    # order matters: clean -> engineer -> drop ids -> encode -> align -> impute -> scale
    df = clean_data(df)
    df = engineer_features(df)

    # grab ids after cleaning so length matches X exactly
    ids = df['encounter_id'].copy() if 'encounter_id' in df.columns else pd.Series(range(len(df)))

    # drop target and id columns
    df = df.drop(columns=[TARGET, 'readmitted', 'encounter_id', 'patient_nbr'], errors='ignore')

    # one-hot encode any leftover object columns — same as train.py
    obj_cols = df.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        print(f'  one-hot encoding: {obj_cols}')
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # force numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # align columns to exactly what the model was trained on
    df = df.reindex(columns=feature_cols, fill_value=0)

    # apply the saved imputer and scaler — must use transform not fit_transform
    # fitting on test data would be data leakage
    X = imputer.transform(df)
    X = scaler.transform(X)

    print(f'  features after alignment: {df.shape[1]}')
    return X, ids


def predict(model, X, ids):
    # keras returns shape (n, 1) so flatten to 1d
    proba = model.predict(X, verbose=0).flatten()
    preds = (proba >= 0.5).astype(int)

    # confidence = how far the probability is from 0.5
    # 1.0 = completely certain, 0.0 = totally uncertain
    confidence = np.abs(proba - 0.5) * 2

    results = pd.DataFrame({
        'id': ids.values,
        'prediction': preds,
        'probability': np.round(proba, 4),
        'confidence': np.round(confidence, 4),
    })

    return results


def main():
    print('=' * 50)
    print('  Model 2: Readmission Risk — DNN Predict')
    print('=' * 50)

    # 1. load saved model and all preprocessing artifacts
    model, feature_cols, imputer, scaler = load_artifacts()

    # 2. load raw test data
    df = load_test_data()

    # 3. run through the same preprocessing pipeline as training
    print('\npreparing test data...')
    X, ids = prepare_test_data(df, feature_cols, imputer, scaler)

    # 4. generate predictions
    print('\nrunning predictions...')
    results = predict(model, X, ids)

    # 5. save output to test_data/model2_results.csv
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print(f'\nresults saved → {OUTPUT_PATH}')
    print(f'total predictions : {len(results)}')
    print(f'predicted readmit : {results["prediction"].sum()} ({results["prediction"].mean()*100:.1f}%)')
    print('\n✅ model 2 prediction complete.')


if __name__ == '__main__':
    main()
