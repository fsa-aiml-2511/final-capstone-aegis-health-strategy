#!/usr/bin/env python3
"""
Model 5: Innovation — Training Script
=======================================
This is your team's innovation model. Identify a problem in the data that
Models 1-4 don't address, and build a model that solves it.

Requirements:
- Clear value proposition (why does this model matter?)
- Defined success metric (you choose what to optimize)
- Cost-benefit estimate (what's the ROI?)

Use whatever approach fits your problem best — traditional ML, deep learning,
clustering, anomaly detection, time series, etc.
"""
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"
UNPROCESSED_DATA = PROJECT_ROOT / "data" / "raw"
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"

from pipelines.data_pipeline import clean_data_model_5, engineer_features_model_5


def create_processed_data():
    #function created so it can create the processed data from the raw for cleaner model code
    filepath = UNPROCESSED_DATA / "patient_encounters_2023.csv"
    filepath_2 = PROCESSED_DATA / "patient_encounters_2023_processed.csv"
    df = pd.read_csv(filepath)

    df = df.copy()

    df = clean_data_model_5(df)
    df = engineer_features_model_5(df)

    df = df.drop(columns=["encounter_id"], errors="ignore")

    df.to_csv(filepath_2, index=False)

    



def load_data():
    """Load data for your innovation model."""
    
    #pull the processed data
    filepath = PROCESSED_DATA / "patient_encounters_2023_processed.csv"
    return pd.read_csv(filepath)


def preprocess(df):
    """Preprocess data for your chosen problem."""
    from sklearn.model_selection import train_test_split

    #make df copy
    df = df.copy()

    #set target and features and drop diabetesmed from features
    y = df["diabetesMed"]
    X = df.drop(columns=["diabetesMed", "encounter_id"], errors="ignore")

    #train test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_val, y_train, y_val




def train_model(X_train, y_train):
    """Train an XGBoost classifier."""
    from xgboost import XGBClassifier

    
    """from sklearn.utils.class_weight import compute_sample_weight"""

    """sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )"""

    #trains xgboost model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42 
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate with your custom success metric.

    Must include:
    - Your chosen metric (and why you chose it)
    - Baseline comparison (what would a naive approach get?)
    - Business impact estimate
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    
    

    #get probability scores for class 1, needs medication and allows us to use a threshold
    y_probs = model.predict_proba(X_val)[:, 1]

    # Custom threshold chosen after testing multiple values
    # 0.70 gave the best balance between precision and recall
    threshold = 0.70

    # Convert probabilities into final predictions using custom threshold
    y_pred = (y_probs >= threshold).astype(int)

    #scores
    # Standard accuracy score
    accuracy = accuracy_score(y_val, y_pred)
    
    # Binary F1 score (focuses on positive class only)
    # Useful for measuring how well we predict medication need
    f1 = f1_score(y_val, y_pred)

    # Macro F1 score (treats both classes equally)
    # Best for measuring class balance
    f1_macro = f1_score(y_val, y_pred, average="macro")

    # Weighted F1 score (accounts for class imbalance)
    f1_weighted = f1_score(y_val, y_pred, average="weighted")

    # Naive baseline accuracy (predicting only the majority class)
    # Used to compare if the model adds value over a simple guess
    baseline = max(y_val.mean(), 1 - y_val.mean())

    # Print evaluation results
    print("=== Innovation Model Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Naive baseline accuracy: {baseline:.4f}")
    
    # Print detailed per-class performance metrics
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

     # Explain why F1 was selected
    print("\nChosen success metric: F1 Score")
    print("It balances precision and recall for identifying patients likely")
    print("to require diabetes medication.")

    # Explain business value
    print("\nValue proposition:")
    print("This model helps estimate diabetes medication need using non-medication")
    print("clinical features, which could support early intervention and care planning.")

    # Explain potential ROI
    print("\nCost-benefit estimate:")
    print("If providers can identify likely medication need earlier, they may improve")
    print("care coordination and reduce delays in treatment decisions.")


def save_model(model):
    """Save your model to saved_model/.

    Example:
        import joblib
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    """
    import joblib

    #saves model
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")


def main():
    
    # 0.  creates the processed data
    create_processed_data()
    
    # 1. Load the processed data
    df = load_data()

    # 2. Preprocess

    X_train, X_val, y_train, y_val = preprocess(df)

    # 3. Train baseline (work was done seperately to determine that this was best model)
    #i evaluated decision tree, random forest, xgboost, and logistic regression, and xgboost was the best
    
    model = train_model(X_train, y_train)

    # 4. Evaluate baseline

    evaluate_model(model, X_val, y_val)
    
    # 5. Save

    save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
