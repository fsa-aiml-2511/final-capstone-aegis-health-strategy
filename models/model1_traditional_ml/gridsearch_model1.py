"""
Model 1: XGBoost Grid Search
=============================
Runs an exhaustive grid search over XGBoost hyperparameters to find
the best combination for AUC-ROC. Takes a while — go get a coffee.

Usage (from project root):
    python models/model1_traditional_ml/gridsearch_model1.py

Prints the best params at the end — copy them into train.py manually.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline import (
    load_raw_data,
    clean_data,
    engineer_features,
    split_data,
)

TARGET = 'readmission_binary'


def load_and_prepare():
    print('loading and preparing data...')
    df = load_raw_data('patient_encounters_2023.csv')
    df = clean_data(df)
    df = engineer_features(df)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, 'encounter_id'], errors='ignore')

    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors='coerce')
    print(f'  features: {X.shape[1]}  |  rows: {X.shape[0]}')
    return X, y


def main():
    print('=' * 55)
    print('  Model 1: XGBoost Grid Search')
    print('=' * 55)

    X, y = load_and_prepare()
    X_train, X_val, y_train, y_val = split_data(X, y)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos

    # pipeline with just imputer + XGBoost
    # no scaler needed for tree models
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', XGBClassifier(
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )),
    ])

    # grid of params to search
    # prefixed with 'clf__' because they're inside the pipeline
    param_grid = {
        'clf__n_estimators':      [500, 700, 1000],
        'clf__max_depth':         [4, 5, 6],
        'clf__learning_rate':     [0.01, 0.03, 0.05],
        'clf__subsample':         [0.8, 0.9],
        'clf__colsample_bytree':  [0.8, 0.9],
        'clf__min_child_weight':  [3, 5],
        'clf__gamma':             [0.0, 0.1],
        'clf__reg_alpha':         [0.0, 0.1],
        'clf__reg_lambda':        [1.0, 1.5],
    }

    # stratified k-fold keeps class balance across folds
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    print(f'\nstarting grid search...')
    print(f'this will try a LOT of combinations — be patient\n')

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,       # use all CPU cores
        verbose=2,       # print progress
        refit=True,      # refit best model on full training data
    )

    grid_search.fit(X_train, y_train)

    # evaluate best model on held-out val set
    best_model = grid_search.best_estimator_
    val_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    print(f"\n{'=' * 55}")
    print(f'  grid search complete')
    print(f'  best CV AUC  : {grid_search.best_score_:.4f}')
    print(f'  val AUC      : {val_auc:.4f}')
    print(f"  benchmark    : {'✅ met (>0.70)' if val_auc >= 0.70 else '❌ below 0.70'}")
    print(f'{"=" * 55}')
    print(f'\n  best params (copy these into train.py):')
    for param, value in grid_search.best_params_.items():
        # strip the 'clf__' prefix for readability
        clean_name = param.replace('clf__', '')
        print(f'    {clean_name}: {value}')
    print(f'{"=" * 55}')


if __name__ == '__main__':
    main()
