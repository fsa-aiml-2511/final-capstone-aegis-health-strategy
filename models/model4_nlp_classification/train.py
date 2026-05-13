#!/usr/bin/env python3
"""
Model 4: NLP Classification - Training Script
Classifies patient drug reviews into 3 effectiveness categories.
"""

import re
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# paths
RAW_DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "healthcare_csvs" / "patient_medication_feedback.csv"
SAVE_DIR = Path(__file__).resolve().parent / "saved_model"


# map the 5 raw effectiveness labels down to 3
EFFECTIVENESS_MAP = {
    'Highly Effective': 'Highly Effective',
    'Considerably Effective': 'Somewhat Effective',
    'Moderately Effective': 'Somewhat Effective',
    'Marginally Effective': 'Somewhat Effective',
    'Ineffective': 'Ineffective',
}


def clean_text(text):
    """basic text cleaning for review strings"""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = text.lower()
    # strip html-ish stuff
    text = re.sub(r'<.*?>', ' ', text)
    # get rid of urls
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # unicode junk like %u2019 etc
    text = re.sub(r'%u[0-9a-fA-F]{4}', ' ', text)
    # keep letters, numbers, spaces only
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prep():
    """load the raw csv and get it ready for training"""
    df = pd.read_csv(RAW_DATA)

    # map effectiveness to 3 classes
    df['label'] = df['effectiveness'].str.strip().map(EFFECTIVENESS_MAP)

    # drop anything that didn't map (shouldn't happen but just in case)
    df = df.dropna(subset=['label'])

    # benefitsReview is the only text column with actual data
    # sideEffectsReview and commentsReview are entirely null
    df['clean_text'] = df['benefitsReview'].apply(clean_text)

    # drop empty reviews
    df = df[df['clean_text'].str.len() > 0].copy()

    return df


def main():
    print("loading data...")
    df = load_and_prep()
    print(f"  {len(df)} reviews after cleaning")
    print(f"  class distribution:\n{df['label'].value_counts().to_string()}\n")

    X = df['clean_text']
    y = df['label']

    # stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # tfidf - unigrams and bigrams, cap at 50k features
    print("fitting tfidf...")
    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)

    # logistic regression w/ balanced class weights since highly effective dominates
    print("training model...")
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        C=1.0,
        solver='lbfgs',
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    # evaluate
    y_pred = model.predict(X_val_vec)
    print("\n--- Validation Results ---")
    print(classification_report(y_val, y_pred))

    wf1 = f1_score(y_val, y_pred, average='weighted')
    acc = (y_pred == y_val.values).mean()
    print(f"accuracy:    {acc:.4f}")
    print(f"weighted f1: {wf1:.4f}")

    print("\nconfusion matrix:")
    labels = ['Highly Effective', 'Somewhat Effective', 'Ineffective']
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # save everything
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVE_DIR / "model.joblib")
    joblib.dump(tfidf, SAVE_DIR / "tfidf_vectorizer.joblib")
    print(f"\nmodel and vectorizer saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
