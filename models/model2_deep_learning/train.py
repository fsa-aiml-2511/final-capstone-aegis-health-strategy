"""
Model 2: Readmission Risk — Deep Neural Network — Train
========================================================
Builds and trains a Keras DNN on the same encounter data as Model 1.
Compares DNN performance against Model 1's XGBoost result.

Usage (from project root):
    python models/model2_deep_learning/train.py

Saves:
    models/model2_deep_learning/saved_model/model.keras
    models/model2_deep_learning/saved_model/feature_cols.joblib
    models/model2_deep_learning/saved_model/imputer.joblib
    models/model2_deep_learning/saved_model/scaler.joblib
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras import regularizers
warnings.filterwarnings('ignore')

# this file is at models/model2_deep_learning/train.py
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

# model 1 AUC for comparison at the end
MODEL1_AUC = 0.6909


def load_and_prepare():
    # exact same pipeline as model 1 — same data, same features
    print('loading raw data...')
    df = load_raw_data('patient_encounters_2023.csv')
    print(f'  raw shape: {df.shape}')

    df = clean_data(df)
    df = engineer_features(df)
    print(f'  processed shape: {df.shape}')

    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found — check that 'readmitted' exists in raw data")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, 'encounter_id'], errors='ignore')

    # one-hot encode any leftover object columns
    obj_cols = X.select_dtypes(include='object').columns.tolist()
    if obj_cols:
        print(f'  one-hot encoding: {obj_cols}')
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    X = X.apply(pd.to_numeric, errors='coerce')

    print(f'  features: {X.shape[1]}')
    print(f'  class balance — 0: {(y==0).sum()}  1: {(y==1).sum()}')
    return X, y


def preprocess(X_train, X_val):
    # DNNs are sensitive to feature scale — impute then standardize
    # we save the imputer and scaler so predict.py applies the exact same transform
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)

    return X_train_scaled, X_val_scaled, imputer, scaler


def build_model(input_dim):
    # architecture: input -> dense layers with batch norm and dropout -> sigmoid output
    # batch norm stabilizes training, dropout prevents overfitting
    # getting progressively smaller (256 -> 128 -> 64 -> 32) is a standard funnel pattern

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),

        # sigmoid output for binary classification probability
        layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['AUC'],
    )

    return model


def train_model(model, X_train, y_train, X_val, y_val):
    # class weight to handle imbalance — same idea as scale_pos_weight in XGBoost
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos if pos > 0 else 1.0}
    print(f'  class weight: {class_weight}')

    # stop if val AUC doesnt improve for 20 epochs now.
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    # cut learning rate in half if val AUC plateaus for 5 epochs (5 didnt work well, lets try 10 )
    #
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=5,
        mode='max',
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=128,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return history


def evaluate(model, X_val, y_val):
    # keras predict returns shape (n, 1) so flatten to 1d
    proba = model.predict(X_val, verbose=0).flatten()
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_val, proba)
    report = classification_report(y_val, preds, output_dict=True)

    print(f"\n  Deep Neural Network")
    print(f"  AUC-ROC  : {auc:.4f}  {'✅' if auc >= 0.70 else '❌ below 0.70 benchmark'}")
    print(f"  F1 (1)   : {report['1']['f1-score']:.4f}")
    print(f"  Precision: {report['1']['precision']:.4f}  Recall: {report['1']['recall']:.4f}")

    return auc


def main():
    print('=' * 50)
    print('  Model 2: Readmission Risk — DNN')
    print('=' * 50)

    # 1. load and prepare — identical to model 1
    X, y = load_and_prepare()
    feature_cols = X.columns.tolist()

    # 2. train/val split
    X_train, X_val, y_train, y_val = split_data(X, y)
    print(f'\ntrain: {len(X_train):,}  |  val: {len(X_val):,}')

    # 3. scale features — critical for DNNs, not needed for tree models
    print('\nscaling features...')
    X_train_scaled, X_val_scaled, imputer, scaler = preprocess(X_train, X_val)

    # 4. build and summarize the model
    model = build_model(input_dim=X_train_scaled.shape[1])
    model.summary()

    # 5. train
    print('\ntraining DNN...')
    train_model(model, X_train_scaled, y_train.values, X_val_scaled, y_val.values)

    # 6. evaluate
    dnn_auc = evaluate(model, X_val_scaled, y_val)

    # 7. compare against model 1
    print(f"\n{'=' * 50}")
    print(f"  model comparison")
    print(f"  Model 1 XGBoost : {MODEL1_AUC:.4f}")
    print(f"  Model 2 DNN     : {dnn_auc:.4f}")
    diff = dnn_auc - MODEL1_AUC
    if diff > 0:
        print(f"  DNN is better by {diff:.4f} — deep learning wins on this dataset")
    else:
        print(f"  XGBoost is better by {abs(diff):.4f} — traditional ML wins on this dataset")
    print(f"  benchmark  : {'✅ met (>0.70)' if dnn_auc >= 0.70 else '❌ below 0.70'}")
    print(f"  stretch    : {'✅ met (>0.80)!' if dnn_auc >= 0.80 else 'not yet (>0.80)'}")
    print('=' * 50)

    # 8. save everything predict.py needs
    # scaler and imputer must be saved so predict.py applies the exact same transform
    model.save(SAVED_MODEL_DIR / 'model.keras')
    joblib.dump(feature_cols, SAVED_MODEL_DIR / 'feature_cols.joblib')
    joblib.dump(imputer, SAVED_MODEL_DIR / 'imputer.joblib')
    joblib.dump(scaler, SAVED_MODEL_DIR / 'scaler.joblib')

    print(f'\nmodel saved        → {SAVED_MODEL_DIR / "model.keras"}')
    print(f'feature cols saved → {SAVED_MODEL_DIR / "feature_cols.joblib"}')
    print(f'imputer saved      → {SAVED_MODEL_DIR / "imputer.joblib"}')
    print(f'scaler saved       → {SAVED_MODEL_DIR / "scaler.joblib"}')
    print('\n✅ model 2 training complete.')


if __name__ == '__main__':
    main()
