# Model 4: NLP Text Classification

Classifies a patient's written review of a medication into one of three
effectiveness categories.

## What this model does

Given a free-text patient review describing the **benefits** of a medication,
the model predicts one of:

- **Highly Effective**
- **Somewhat Effective**
- **Ineffective**

It was trained on the `benefitsReview` column of
`data/raw/healthcare_csvs/patient_medication_feedback.csv` — real patient
narratives describing whether a drug helped them.

## How to use it in the web app

In the Streamlit app, choose **"Model 4: NLP (Text Classification)"** from the
sidebar, type a patient review into the text box, and click **Classify**.

### What to type in the box

A **patient review** — a sentence or two written the way a real person would
describe their experience with a medication. That's the kind of text the model
was trained on, so that's the only kind of input where the prediction is
meaningful.

### What *not* to type (and why your test inputs gave weird results)

- **A single drug name** (e.g. `"Hydromorphone"`).
  The model is not a drug lookup — it has never seen drug names as inputs
  during training. It only ever saw review *text*. Typing a drug name gives
  the model nothing useful to classify, so it falls back to whatever weak
  signal it can find and returns a prediction anyway.

- **Random words** (e.g. `"hello"`).
  The model **always** returns one of its three classes — that's how
  classifiers work. It picks whichever class is the "least bad fit" for the
  input. So `"hello"` returns a prediction, but it's meaningless: the model
  is essentially guessing.

- **Empty / very short input.**
  Same issue as above. The TF-IDF vectorizer turns text into a vector of
  numbers based on which words appear. With only one or two words there's
  almost no signal, so the prediction is essentially noise.

> **Important:** The "Confidence" score is *not* an "is this a real review?"
> check. It's just how sure the model is between its three labels. The model
> can be very confident on garbage input — that doesn't mean the input was
> meaningful, only that one of the three classes happened to score higher
> than the others.

### Example inputs that work

| Input | Expected prediction |
|---|---|
| *"This medication completely cleared up my symptoms within a week and I had no issues. I feel like myself again."* | Highly Effective |
| *"Took it for two months and saw a small improvement but the symptoms never fully went away."* | Somewhat Effective |
| *"Did not work for me at all. My pain was the same after finishing the prescription."* | Ineffective |

You can also copy any value from the `benefitsReview` column of
`patient_medication_feedback.csv` directly into the text box.

## How it works under the hood

1. **Text cleaning** (`clean_text` in `train.py` / `predict.py`)
   - Lowercase the text.
   - Expand contractions (`"didn't"` → `"did not"`) — done *before* stripping
     punctuation so negations survive. This matters a lot: `"did not help"`
     and `"did help"` should give very different predictions.
   - Strip HTML tags, URLs, unicode escapes (`%u2019` etc.), and any
     non-letter / non-digit characters.
   - Collapse whitespace.

2. **Vectorization** (TF-IDF)
   - Word n-grams (1–2) — captures phrases like `"did not work"`.
   - Character n-grams (3–5) — helps with misspellings and rare words.
   - Both are combined via `FeatureUnion`.

3. **Classification** (Logistic Regression)
   - Trained with `class_weight='balanced'` because the three classes are
     unevenly represented in the data.
   - The model outputs a probability for each of the three classes; the
     highest-probability class is the prediction, and that probability is
     reported as the confidence.

4. **Label mapping** — the raw dataset has 5 effectiveness labels which are
   collapsed to 3 (see `EFFECTIVENESS_MAP` in `train.py`):
   - `Highly Effective` → **Highly Effective**
   - `Considerably Effective`, `Moderately Effective`, `Marginally Effective`
     → **Somewhat Effective**
   - `Ineffective` → **Ineffective**

## Files

- `train.py` — trains the model, runs a small hyperparameter sweep
  (word-only vs word+char TF-IDF, several `C` values), prints a classification
  report and confusion matrix on the validation split, and saves the best
  model + vectorizer to `saved_model/`.
- `predict.py` — loads the saved model and runs predictions on a CSV in
  `test_data/`, writing results to `test_data/model4_results.csv`.
- `saved_model/model.joblib` — the trained logistic regression model.
- `saved_model/tfidf_vectorizer.joblib` — the fitted TF-IDF vectorizer.
  Must be loaded together with the model — they're a matched pair.

## Retraining

```bash
python models/model4_nlp_classification/train.py
```

This re-runs the full training pipeline and overwrites `saved_model/`.

## Limitations to be aware of

- **The model classifies effectiveness, not sentiment in general.** It only
  produces meaningful predictions on text that resembles a patient review of
  a medication's benefits.
- **No "unknown / not a review" output.** The classifier always returns one
  of three classes. If you need to filter out non-review inputs, that has to
  happen *before* the text reaches the model (e.g. minimum word count, or a
  separate "is this a review?" classifier).
- **Trained only on the `benefitsReview` column** — `sideEffectsReview` and
  `commentsReview` were entirely null in the source data, so the model has
  not learned anything about side effects or general comments.
