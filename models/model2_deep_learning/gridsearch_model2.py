"""
Model 2: DNN Hyperparameter Search
=====================================
Tries different DNN architectures and training configs to find
the best combination for AUC-ROC. Uses a manual loop since Keras
doesn't work cleanly with sklearn's GridSearchCV.

Usage (from project root):
    python models/model2_deep_learning/gridsearch_model2.py

Prints the best config at the end — copy it into train.py manually.
"""

import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

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


def preprocess(X_train, X_val):
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    return X_train_scaled, X_val_scaled


def build_model(input_dim, units, dropout_rate, learning_rate, l2_reg):
    # builds a funnel architecture based on the config passed in
    # units is a list like [256, 128, 64] defining each hidden layer
    model_layers = [layers.Input(shape=(input_dim,))]

    for u in units:
        model_layers.append(
            layers.Dense(u, activation='relu',
                        kernel_regularizer=regularizers.l2(l2_reg))
        )
        model_layers.append(layers.BatchNormalization())
        model_layers.append(layers.Dropout(dropout_rate))

    model_layers.append(layers.Dense(1, activation='sigmoid'))

    model = keras.Sequential(model_layers)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['AUC'],
    )
    return model


def train_and_evaluate(model, X_train, y_train, X_val, y_val, batch_size):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos}

    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=0       # quiet during grid search
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        mode='max',
        min_lr=1e-6,
        verbose=0
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=0       # quiet during grid search
    )

    proba = model.predict(X_val, verbose=0).flatten()
    auc = roc_auc_score(y_val, proba)
    return auc


def main():
    print('=' * 55)
    print('  Model 2: DNN Hyperparameter Search')
    print('=' * 55)

    X, y = load_and_prepare()
    X_train, X_val, y_train, y_val = split_data(X, y)

    print('\nscaling features...')
    X_train_scaled, X_val_scaled = preprocess(X_train, X_val)
    input_dim = X_train_scaled.shape[1]

    # configs to try — each combination will be trained and evaluated
    param_grid = {
        'units':         [[256, 128, 64], [128, 64, 32], [256, 128, 64, 32]],
        'dropout_rate':  [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005],
        'l2_reg':        [0.0, 0.001, 0.01],
        'batch_size':    [256, 512],
    }

    # generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    total = len(combinations)

    print(f'\ntrying {total} configurations...\n')

    best_auc = 0
    best_config = None
    results = []

    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))

        # clear keras session between runs to avoid memory buildup
        tf.keras.backend.clear_session()

        model = build_model(
            input_dim=input_dim,
            units=config['units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            l2_reg=config['l2_reg'],
        )

        auc = train_and_evaluate(
            model,
            X_train_scaled, y_train.values,
            X_val_scaled, y_val.values,
            batch_size=config['batch_size'],
        )

        results.append((auc, config))

        if auc > best_auc:
            best_auc = auc
            best_config = config

        print(f'  [{i+1}/{total}]  AUC: {auc:.4f}  '
              f'units={config["units"]}  '
              f'dropout={config["dropout_rate"]}  '
              f'lr={config["learning_rate"]}  '
              f'l2={config["l2_reg"]}  '
              f'batch={config["batch_size"]}  '
              f'{"← best so far!" if auc == best_auc else ""}')

    # sort results and show top 5
    results.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{'=' * 55}")
    print(f'  search complete')
    print(f'  best val AUC : {best_auc:.4f}')
    print(f"  benchmark    : {'✅ met (>0.70)' if best_auc >= 0.70 else '❌ below 0.70'}")
    print(f'\n  top 5 configurations:')
    for auc, config in results[:5]:
        print(f'    AUC: {auc:.4f}  {config}')
    print(f'\n  best config (copy into train.py):')
    for k, v in best_config.items():
        print(f'    {k}: {v}')
    print(f'{"=" * 55}')


if __name__ == '__main__':
    main()
