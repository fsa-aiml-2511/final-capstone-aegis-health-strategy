"""
Model 1: Readmission Risk — Traditional ML — Train
====================================================
Trains XGBoost, Random Forest, and Logistic Regression on encounter data.
Picks the best model by AUC-ROC and saves it along with the feature column
list so predict.py can align test data to the exact same columns.

Usage (from project root):
    python models/model1_traditional_ml/train.py

Saves:
    models/model1_traditional_ml/saved_model/model.joblib
    models/model1_traditional_ml/saved_model/feature_cols.joblib
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print('xgboost not installed — skipping XGBoost')

warnings.filterwarnings('ignore')

# this file is at models/model1_traditional_ml/train.py
# so project root is 2 levels up
MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    load_raw_data,
    clean_data,
    engineer_features,
    split_data,
)

SAVED_MODEL_DIR = MODEL_DIR / 'saved_model'
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = 'readmission_binary'


def load_and_prepare():
    # load raw csv, run the full pipeline, return X and y
    print('loading raw data...')
    df = load_raw_data('patient_encounters_2023.csv')
    print(f'  raw shape: {df.shape}')

    df = clean_data(df)
    df = engineer_features(df)
    print(f'  processed shape: {df.shape}')

    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found — check that 'readmitted' exists in the raw data")

    y = df[TARGET].astype(int)

    # drop the target and IDs — encounter_id is kept through the pipeline
    # and used as the output 'id' column in predict.py, so drop it from features here
    X = df.drop(columns=[TARGET, 'encounter_id'], errors='ignore')

    # catch any leftover object columns and one-hot encode them
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        print(f'  one-hot encoding leftover object cols: {obj_cols}')
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # make sure everything is numeric — coerce anything that slipped through
    X = X.apply(pd.to_numeric, errors='coerce')

    print(f'  features: {X.shape[1]}')
    print(f'  class balance — 0: {(y==0).sum()}  1: {(y==1).sum()}')
    return X, y


def build_models(scale_pos_weight):
    # returns a dict of model name -> sklearn pipeline
    # classes are nearly balanced (~52/47) so we use scale_pos_weight for XGBoost
    # and class_weight='balanced' for logistic regression and random forest

    models = {}

    models['Logistic Regression'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )),
    ])

    models['Random Forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )),
    ])

    if XGBOOST_AVAILABLE:
        # lower learning rate + more trees = more careful learning
        # max_depth=6 gives more room to capture feature interactions
        # reg_alpha/lambda regularize to avoid overfitting with the extra depth
        # scale_pos_weight handles mild class imbalance without SMOTE
        models['XGBoost'] = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', XGBClassifier(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                min_child_weight=5,
                gamma=0.05,
                reg_alpha=0.05,
                reg_lambda=1.5,
                scale_pos_weight=scale_pos_weight,
                eval_metric='auc',
                random_state=42,
                n_jobs=-1
            )),
        ])

    return models


def evaluate(model, X_val, y_val, name):
    # print AUC-ROC and classification report for a fitted model
    proba = model.predict_proba(X_val)[:, 1]
    preds = model.predict(X_val)
    auc = roc_auc_score(y_val, proba)
    report = classification_report(y_val, preds, output_dict=True)

    print(f"\n  {name}")
    print(f"  AUC-ROC  : {auc:.4f}  {'benchmark reached!' if auc >= 0.70 else ' below 0.70 benchmark'}")
    print(f"  F1 (1)   : {report['1']['f1-score']:.4f}")
    print(f"  Precision: {report['1']['precision']:.4f}  Recall: {report['1']['recall']:.4f}")

    return auc


def main():
    print('=' * 50)
    print('  Model 1: Readmission Risk — Traditional ML')
    print('=' * 50)

    # 1. load and prepare
    X, y = load_and_prepare()
    feature_cols = X.columns.tolist()

    # 2. train/val split — stratified to preserve class balance
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f'\ntrain: {len(X_train):,}  |  val: {len(X_val):,}')

    # scale_pos_weight = ratio of negatives to positives
    # tells XGBoost how much to up-weight the minority class
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f'scale_pos_weight: {scale_pos_weight:.2f}')

    # 3. train all models and track AUC
    models = build_models(scale_pos_weight)
    results = {}

    for name, model in models.items():
        print(f'\ntraining {name}...')
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_val, y_val, name)

    # 4. pick the best model by AUC-ROC
    best_name = max(results, key=lambda k: results[k])
    best_model = models[best_name]
    best_auc = results[best_name]

    print(f"\n{'=' * 50}")
    print(f"  best model : {best_name}")
    print(f"  AUC-ROC    : {best_auc:.4f}")
    print(f"  benchmark  : {'Benchmark met (>0.70)' if best_auc >= 0.70 else ' below 0.70'}")
    print(f"  stretch    : {'Stretch Goal met (>0.80)!' if best_auc >= 0.80 else 'not yet (>0.80)'}")
    print('=' * 50)

    # 5. save the model and the feature column list
    # feature_cols is critical — predict.py uses it to reindex test data
    # to the exact same columns the model was trained on
    joblib.dump(best_model, SAVED_MODEL_DIR / 'model.joblib')
    joblib.dump(feature_cols, SAVED_MODEL_DIR / 'feature_cols.joblib')

    print(f'\nmodel saved        → {SAVED_MODEL_DIR / "model.joblib"}')
    print(f'feature cols saved → {SAVED_MODEL_DIR / "feature_cols.joblib"}')
    print('\nGoalpost! model 1 training complete.')


if __name__ == '__main__':
    main()
