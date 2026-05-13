"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(filename):
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "patient_encounters_2023.csv")

    Returns:
        pandas DataFrame

    Example:
        df = load_raw_data("patient_encounters_2023.csv")
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    # TODO: Load and return the CSV
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Apply common data cleaning steps.

    Things to handle:
    - Missing value encoding (e.g., '?' -> NaN)
    - Data type conversions
    - Remove duplicates
    - Drop irrelevant columns

    Returns:
        Cleaned DataFrame
    """
    # TODO: Add your cleaning logic

    df = df.copy()
   
    
    #drop duplicates
    if 'encounter_id' in df.columns:
        df = df.drop_duplicates(subset=['encounter_id'])

    #Replace '?' with NaN across the dataframe
    df = df.replace('?', np.nan)

    #Convert age brackets to numeric midpoint
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].map(age_map)

    #Exclude death/hospice discharges before creating readmission target
    exclude_dispositions = [11, 13, 14, 19, 20, 21]
    if 'discharge_disposition_id' in df.columns:
        df = df[
            ~df['discharge_disposition_id'].isin(exclude_dispositions)
        ].copy()

    #Create diagnosis category columns
    diag_cols = ['diag_1', 'diag_2', 'diag_3']

    for col in diag_cols:
        if col in df.columns:
            code_str = df[col].astype(str).str.strip()

            code_num = pd.to_numeric(
                code_str.str.extract(r'^(\d{3})')[0],
                errors='coerce'
            )

            df[f'{col}_category'] = np.select(
                [
                    df[col].isna(),
                    code_str.str.startswith('250'),
                    code_num.between(390, 459),
                    code_num.between(460, 519),
                    code_num.between(520, 579),
                    code_str.str.startswith(('V', 'E'))
                ],
                [
                    'missing',
                    'diabetes',
                    'circulatory',
                    'respiratory',
                    'digestive',
                    'external'
                ],
                default='other'
            )



    return df



def engineer_features(df):
    """Create new features from existing columns.
    Assumes clean_data() has already run first.

    Examples:
    - Parse datetime columns -> hour, day_of_week, month
    - Create binary flags from categorical data
    - Bin continuous variables into categories
    - Interaction features

    Returns:
        DataFrame with new feature columns
    """
    # TODO: Add your feature engineering

    df = df.copy()



    #Create binary readmission target
    if 'readmitted' in df.columns:
        df['readmission_binary'] = (df['readmitted'] != 'NO').astype(int)

    #Medication-derived features MUST happen before medication encoding
    med_cols_all = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_cols_all = [c for c in med_cols_all if c in df.columns]

    if med_cols_all:
        # Count medications that are active
        df['num_active_meds'] = (df[med_cols_all] != 'No').sum(axis=1)

        #count how many meds were changed
        df['n_meds_changed'] = df[med_cols_all].apply(
            lambda row: sum(1 for v in row if v in ['Up', 'Down']),
            axis=1
        )

        # Count medications increased
        df['n_meds_increased'] = df[med_cols_all].apply(
            lambda row: sum(1 for v in row if v == 'Up'),
            axis=1
        )

    #Clinical complexity score
    complexity_parts = {
        'num_lab_procedures': 50,
        'num_procedures': 6,
        'num_medications': 20,
        'number_diagnoses': 9,
        'number_emergency': 3,
        'number_inpatient': 5
    }

    if all(col in df.columns for col in complexity_parts):
        df['complexity_score'] = sum(
            df[col].fillna(0) / divisor
            for col, divisor in complexity_parts.items()
        )

    #ordinal encode max glu and a1c, and create a tested column for each as well
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum_tested'] = df['max_glu_serum'].notna().astype(int)
        df['max_glu_serum'] = df['max_glu_serum'].map({
            'Norm': 0,
            '>200': 1,
            '>300': 2
        }).fillna(-1)

    if 'A1Cresult' in df.columns:
        df['A1Cresult_tested'] = df['A1Cresult'].notna().astype(int)
        df['A1Cresult'] = df['A1Cresult'].map({
            'Norm': 0,
            '>7': 1,
            '>8': 2
        }).fillna(-1)

    #Race: drop missing and other, then one-hot encode
    if 'race' in df.columns:
        df = df.dropna(subset=['race']).copy()
        df = df[df['race'] != 'Other'].copy()
        df = pd.get_dummies(df, columns=['race'])

    #Gender: drop unknown/invalid, map Male/Female
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Unknown/Invalid'].copy()
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    #Replace age with cleaned numeric age
    if 'age_numeric' in df.columns:
        df['age'] = df['age_numeric']
        df = df.drop(columns=['age_numeric'], errors='ignore')

    #Weight: binary flag for whether weight was recorded
    if 'weight' in df.columns:
        df['weight'] = (
            df['weight'].notna() & (df['weight'] != '')
        ).astype(int)
        df = df.rename(columns={'weight': 'weight_checked'})

    #Drop columns not being used
    df = df.drop(columns=['payer_code'], errors='ignore')
    df = df.drop(columns=['medical_specialty'], errors='ignore')

    #Use diagnosis category columns created in clean_data(), drop columns, and one hot encode
    diag_pairs = [
        ('diag_1', 'diag_1_category'),
        ('diag_2', 'diag_2_category'),
        ('diag_3', 'diag_3_category')
    ]

    for raw_col, cat_col in diag_pairs:
        if cat_col in df.columns:
            df[raw_col] = df[cat_col]

    
    required_diag_cols = ['diag_1', 'diag_2', 'diag_3']
    if all(col in df.columns for col in required_diag_cols):


        #Drop helper category columns after replacing raw diagnosis cols
        df = df.drop(
            columns=['diag_1_category', 'diag_2_category', 'diag_3_category'],
            errors='ignore'
        )

        #One-hot encode diagnosis categories
        df = pd.get_dummies(
            df,
            columns=['diag_1', 'diag_2', 'diag_3'],
            drop_first=False
        )
        # total prior visits — strong readmission signal
    # total prior visits — strong readmission signal
    visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    if all(col in df.columns for col in visit_cols):
        df['total_prior_visits'] = df[visit_cols].sum(axis=1)
        # drop outpatient after combining — weakest signal, already in total_prior_visits
        df = df.drop(columns=['number_outpatient'], errors='ignore')

    # binary flag for high-risk prior inpatient history
    # 2+ inpatient visits is a strong clinical threshold for readmission risk
    if 'number_inpatient' in df.columns:
        df['high_risk_prior'] = (df['number_inpatient'] >= 2).astype(int)

    # length of stay x inpatient history — long stays + repeat admissions = very high risk
    if 'time_in_hospital' in df.columns and 'number_inpatient' in df.columns:
        df['los_x_inpatient'] = df['time_in_hospital'] * df['number_inpatient']

    # insulin flag — must be before medication encoding or insulin is already numeric
    if 'insulin' in df.columns:
        df['on_insulin'] = (df['insulin'] != 'No').astype(int)

    # high prior inpatient visits AND many medications = high risk combo
    if 'number_inpatient' in df.columns and 'num_medications' in df.columns:
        df['inpatient_x_medications'] = df['number_inpatient'] * df['num_medications']

    
    #Encode medication status after counts were created
    med_status_map = {
        'No': 0,
        'Steady': 1,
        'Up': 2,
        'Down': 2
    }

    for col in med_cols_all:
        if col in df.columns:
            df[col] = df[col].map(med_status_map)

    #Drop near-constant medication columns after any counts/encoding
    df = df.drop(
        columns=['examide', 'citoglipton', 'troglitazone'],
        errors='ignore'
    )

    #Binary encode diabetesMed
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})

    #Binary encode change
    if 'change' in df.columns:
        df['change'] = df['change'].map({'Ch': 1, 'No': 0})

    #drop readmitted since it could cause data leakage
    if 'readmitted' in df.columns:
        df = df.drop(columns=['readmitted'], errors='ignore')

    #drop encounter and patient id
    if 'patient_nbr' in df.columns:
        df = df.drop(columns=['patient_nbr'], errors='ignore')

    #one hot encode admission type,discharge disposition, and admission source id's
    cat_id_cols = [
        'admission_type_id',
        'discharge_disposition_id',
        'admission_source_id'
    ]
    existing_cat_id_cols = [col for col in cat_id_cols if col in df.columns]

    if existing_cat_id_cols:
        df = pd.get_dummies(df, columns=existing_cat_id_cols, drop_first=True)
    


    #binary encode all bool values
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation sets.

    IMPORTANT: Use stratify=y for imbalanced classification tasks.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion for validation (default 0.2)
        random_state: For reproducibility

    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed_data(df, filename):
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "encounters_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename):
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)

import joblib

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model1_traditional_ml" / "saved_model"

def save_pipeline_artifacts(feature_cols, filename="feature_cols.joblib"):
    """Save feature column list so predict.py uses exact same columns as train."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_cols, SAVED_MODEL_DIR / filename)
    print(f"Saved feature cols → {SAVED_MODEL_DIR / filename}")

def load_pipeline_artifacts(filename="feature_cols.joblib"):
    """Load saved feature columns for use in predict.py."""
    return joblib.load(SAVED_MODEL_DIR / filename)


def find_test_csv(test_dir, expected_columns=None, name_hint=None):
    """Find the right test CSV in test_data/ folder.
    
    Args:
        test_dir: Path to test_data/
        expected_columns: List of column names to look for
        name_hint: String that should appear in the filename
    
    Returns:
        Path to the matching CSV file
    """
    test_dir = Path(test_dir)
    candidates = list(test_dir.glob("*.csv"))
    
    # filter out results files
    candidates = [f for f in candidates if 'results' not in f.name.lower()]
    
    if name_hint:
        hinted = [f for f in candidates if name_hint.lower() in f.name.lower()]
        if hinted:
            return hinted[0]
    
    if expected_columns and candidates:
        for f in candidates:
            try:
                cols = pd.read_csv(f, nrows=1).columns.tolist()
                if all(c in cols for c in expected_columns):
                    return f
            except Exception:
                continue
    
    if candidates:
        return candidates[0]
    
    raise FileNotFoundError(f"No suitable test CSV found in {test_dir}")

def clean_data_model_5(df):
    """Apply common data cleaning steps.

    Things to handle:
    - Missing value encoding (e.g., '?' -> NaN)
    - Data type conversions
    - Remove duplicates
    - Drop irrelevant columns

    Returns:
        Cleaned DataFrame
    """
    # TODO: Add your cleaning logic

    df = df.copy()
   
    
    #drop duplicates
    if 'encounter_id' in df.columns:
        df = df.drop_duplicates(subset=['encounter_id'])

    #Replace '?' with NaN across the dataframe
    df = df.replace('?', np.nan)

    #Convert age brackets to numeric midpoint
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    if 'age' in df.columns:
        df['age_numeric'] = df['age'].map(age_map)

    #Exclude death/hospice discharges before creating readmission target
    exclude_dispositions = [11, 13, 14, 19, 20, 21]
    if 'discharge_disposition_id' in df.columns:
        df = df[
            ~df['discharge_disposition_id'].isin(exclude_dispositions)
        ].copy()

    #Create diagnosis category columns
    diag_cols = ['diag_1', 'diag_2', 'diag_3']

    for col in diag_cols:
        if col in df.columns:
            code_str = df[col].astype(str).str.strip()

            code_num = pd.to_numeric(
                code_str.str.extract(r'^(\d{3})')[0],
                errors='coerce'
            )

            df[f'{col}_category'] = np.select(
                [
                    df[col].isna(),
                    code_str.str.startswith('250'),
                    code_num.between(390, 459),
                    code_num.between(460, 519),
                    code_num.between(520, 579),
                    code_str.str.startswith(('V', 'E'))
                ],
                [
                    'missing',
                    'diabetes',
                    'circulatory',
                    'respiratory',
                    'digestive',
                    'external'
                ],
                default='other'
            )



    return df

def engineer_features_model_5(df):
    """Create new features from existing columns.
    Assumes clean_data() has already run first.

    Examples:
    - Parse datetime columns -> hour, day_of_week, month
    - Create binary flags from categorical data
    - Bin continuous variables into categories
    - Interaction features

    Returns:
        DataFrame with new feature columns
    """
    # TODO: Add your feature engineering

    df = df.copy()



    #Create binary readmission target
    if 'readmitted' in df.columns:
        df['readmission_binary'] = (df['readmitted'] != 'NO').astype(int)

    #Medication-derived features MUST happen before medication encoding
    med_cols_all = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    med_cols_all = [c for c in med_cols_all if c in df.columns]

    if med_cols_all:
        # Count medications that are active
        df['num_active_meds'] = (df[med_cols_all] != 'No').sum(axis=1)

        #count how many meds were changed
        df['n_meds_changed'] = df[med_cols_all].apply(
            lambda row: sum(1 for v in row if v in ['Up', 'Down']),
            axis=1
        )

        # Count medications increased
        df['n_meds_increased'] = df[med_cols_all].apply(
            lambda row: sum(1 for v in row if v == 'Up'),
            axis=1
        )

    #Clinical complexity score
    complexity_parts = {
        'num_lab_procedures': 50,
        'num_procedures': 6,
        'num_medications': 20,
        'number_diagnoses': 9,
        'number_emergency': 3,
        'number_inpatient': 5
    }

    if all(col in df.columns for col in complexity_parts):
        df['complexity_score'] = sum(
            df[col].fillna(0) / divisor
            for col, divisor in complexity_parts.items()
        )

    #ordinal encode max glu and a1c, and create a tested column for each as well
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum_tested'] = df['max_glu_serum'].notna().astype(int)
        df['max_glu_serum'] = df['max_glu_serum'].map({
            'Norm': 0,
            '>200': 1,
            '>300': 2
        }).fillna(-1)

    if 'A1Cresult' in df.columns:
        df['A1Cresult_tested'] = df['A1Cresult'].notna().astype(int)
        df['A1Cresult'] = df['A1Cresult'].map({
            'Norm': 0,
            '>7': 1,
            '>8': 2
        }).fillna(-1)

    #Race: drop missing and other, then one-hot encode
    if 'race' in df.columns:
        df = df.dropna(subset=['race']).copy()
        df = df[df['race'] != 'Other'].copy()
        df = pd.get_dummies(df, columns=['race'])

    #Gender: drop unknown/invalid, map Male/Female
    if 'gender' in df.columns:
        df = df[df['gender'] != 'Unknown/Invalid'].copy()
        df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

    #Replace age with cleaned numeric age
    if 'age_numeric' in df.columns:
        df['age'] = df['age_numeric']
        df = df.drop(columns=['age_numeric'], errors='ignore')

    #Weight: binary flag for whether weight was recorded
    if 'weight' in df.columns:
        df['weight'] = (
            df['weight'].notna() & (df['weight'] != '')
        ).astype(int)
        df = df.rename(columns={'weight': 'weight_checked'})

    #Drop columns not being used
    df = df.drop(columns=['payer_code'], errors='ignore')
    df = df.drop(columns=['medical_specialty'], errors='ignore')

    #Use diagnosis category columns created in clean_data(), drop columns, and one hot encode
    diag_pairs = [
        ('diag_1', 'diag_1_category'),
        ('diag_2', 'diag_2_category'),
        ('diag_3', 'diag_3_category')
    ]

    for raw_col, cat_col in diag_pairs:
        if cat_col in df.columns:
            df[raw_col] = df[cat_col]

    
    required_diag_cols = ['diag_1', 'diag_2', 'diag_3']
    if all(col in df.columns for col in required_diag_cols):


        #Drop helper category columns after replacing raw diagnosis cols
        df = df.drop(
            columns=['diag_1_category', 'diag_2_category', 'diag_3_category'],
            errors='ignore'
        )

        #One-hot encode diagnosis categories
        df = pd.get_dummies(
            df,
            columns=['diag_1', 'diag_2', 'diag_3'],
            drop_first=False
        )
        # total prior visits — strong readmission signal
    # total prior visits — strong readmission signal
    visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
    if all(col in df.columns for col in visit_cols):
        df['total_prior_visits'] = df[visit_cols].sum(axis=1)

    # insulin flag — must be before medication encoding or insulin is already numeric
    if 'insulin' in df.columns:
        df['on_insulin'] = (df['insulin'] != 'No').astype(int)

    # high prior inpatient visits AND many medications = high risk combo
    if 'number_inpatient' in df.columns and 'num_medications' in df.columns:
        df['inpatient_x_medications'] = df['number_inpatient'] * df['num_medications']

    
    #Encode medication status after counts were created
    med_status_map = {
        'No': 0,
        'Steady': 1,
        'Up': 2,
        'Down': 2
    }

    for col in med_cols_all:
        if col in df.columns:
            df[col] = df[col].map(med_status_map)

    #Drop near-constant medication columns after any counts/encoding
    df = df.drop(
        columns=['examide', 'citoglipton', 'troglitazone'],
        errors='ignore'
    )

    #Binary encode diabetesMed
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = df['diabetesMed'].map({'Yes': 1, 'No': 0})

    #Binary encode change
    if 'change' in df.columns:
        df['change'] = df['change'].map({'Ch': 1, 'No': 0})

    #drop readmitted since it could cause data leakage
    if 'readmitted' in df.columns:
        df = df.drop(columns=['readmitted'], errors='ignore')

    #drop encounter and patient id
    if 'patient_nbr' in df.columns:
        df = df.drop(columns=['patient_nbr'], errors='ignore')

    #one hot encode admission type,discharge disposition, and admission source id's
    cat_id_cols = [
        'admission_type_id',
        'discharge_disposition_id',
        'admission_source_id'
    ]
    existing_cat_id_cols = [col for col in cat_id_cols if col in df.columns]

    if existing_cat_id_cols:
        df = pd.get_dummies(df, columns=existing_cat_id_cols, drop_first=True)
    


    #binary encode all bool values
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    #drop columns that will cause data leakage in the model
    df = df.drop(columns=['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone',
        'num_active_meds', 'n_meds_changed', 'n_meds_increased',
        'change', 'max_glu_serum_tested', 'A1Cresult_tested', 'complexity_score',
        'on_insulin', 'inpatient_x_medications', 'total_prior_visits', 'num_medications'], errors="ignore")

    return df