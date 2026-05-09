"""
Capstone Web Application
========================
Integrates all 5 models into a single web interface using Streamlit.

Run locally:  streamlit run webapp/app.py
Deploy:       Push to GitHub, then connect to Streamlit Community Cloud
              https://streamlit.io/cloud (free hosting)
"""
import streamlit as st
from pathlib import Path
import pandas as pd
import joblib
from urllib.request import urlretrieve
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Page config
st.set_page_config(
    page_title="Aegis Health Strategy",
    page_icon="⚕️",
    layout="wide",
)

st.title("Aegis Health Strategy")
st.write("")

# Sidebar navigation
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Home",
        "Model 1: Traditional ML",
        "Model 2: Deep Learning",
        "Model 3: CNN (Image Classification)",
        "Model 4: NLP (Text Classification)",
        "Model 5: XGBoost",
        "Our Team" 
    ],
)

# ---------------------------------------------------------------------------
# Helper: Cache model loading so it only happens once
# ---------------------------------------------------------------------------
# Use @st.cache_resource for models — they load once and stay in memory.
#
# Example:
#     @st.cache_resource
#     def load_model1():
#         import joblib
#         return joblib.load("models/model1_traditional_ml/saved_model/model.joblib")
#
#     @st.cache_resource
#     def load_model3():
#         import tensorflow as tf
#         return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")

# ---------------------------------------------------------------------------
# Model pages — fill these in with your model loading and prediction logic
# ---------------------------------------------------------------------------

if model_choice == "Home":
    st.write("Welcome! Use the sidebar to navigate between models.")
    st.markdown("""
## Transforming Healthcare Through Intelligent Analytics

At **Aegis Health Strategy**, we specialize in leveraging artificial intelligence, machine learning, and advanced healthcare analytics to help medical organizations make smarter, faster, and more impactful clinical decisions.

Our mission is to empower healthcare providers with data-driven solutions that improve patient outcomes, optimize operational efficiency, and reduce preventable healthcare costs.

Through innovative predictive modeling, clinical decision support systems, and AI-powered diagnostics, we help healthcare organizations tackle critical challenges such as:

- **Patient Readmission Risk Prediction**  
- **Diabetic Retinopathy Detection**  
- **Medication Effectiveness Analysis**  
- **Clinical Resource Optimization**  
- **Healthcare Data Intelligence**

By combining cutting-edge machine learning models with real-world clinical insight, Aegis Health Strategy bridges the gap between healthcare data and actionable patient care strategies.

### Our Commitment
We believe healthcare innovation should be:
- **Accurate** — delivering reliable predictive insights  
- **Transparent** — providing interpretable AI for clinical trust  
- **Efficient** — streamlining workflows and improving care delivery  
- **Scalable** — built for long-term healthcare transformation  

Explore our platform to discover how intelligent healthcare solutions can drive better decisions, better care, and better outcomes.

**Protecting Patients. Powering Decisions. Advancing Healthcare.**
""")
#model 1 ----
elif model_choice == "Model 1: Traditional ML":
    st.header("Model 1: Readmission Risk — Traditional ML")
    st.write("Predicts whether a diabetic patient will be readmitted to the hospital using XGBoost.")

    @st.cache_resource
    def load_model1():
        model_url = "https://huggingface.co/jfranklingoff/Capstone-Project-Traditional-ML/resolve/main/model.joblib"
        cols_url  = "https://huggingface.co/jfranklingoff/Capstone-Project-Traditional-ML/resolve/main/feature_cols.joblib"

        model_path = Path("model1_model.joblib")
        cols_path  = Path("model1_feature_cols.joblib")

        if not model_path.exists():
            r = requests.get(model_url)
            r.raise_for_status()
            model_path.write_bytes(r.content)

        if not cols_path.exists():
            r = requests.get(cols_url)
            r.raise_for_status()
            cols_path.write_bytes(r.content)

        model = joblib.load(model_path)
        feature_cols = joblib.load(cols_path)
        return model, feature_cols

    with st.spinner("Loading Model 1..."):
        model1, feature_cols1 = load_model1()

    st.subheader("Enter Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_options = {
            "0–10": 5, "10–20": 15, "20–30": 25, "30–40": 35, "40–50": 45,
            "50–60": 55, "60–70": 65, "70–80": 75, "80–90": 85, "90–100": 95
        }
        age_label = st.selectbox("Age Range", list(age_options.keys()), key="m1_age")
        age = age_options[age_label]

        gender = st.selectbox("Gender", ["Male", "Female"], key="m1_gender")
        gender_encoded = 0 if gender == "Male" else 1

        race = st.selectbox("Race", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic"], key="m1_race")

        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=14, value=4, key="m1_los")

        weight_checked = st.selectbox("Weight Recorded?", ["No", "Yes"], key="m1_weight")
        weight_checked = 1 if weight_checked == "Yes" else 0

    with col2:
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=132, value=40, key="m1_labs")
        num_procedures     = st.number_input("Number of Medical Procedures", min_value=0, max_value=6, value=1, key="m1_procs")
        num_medications    = st.number_input("Number of Medications", min_value=1, max_value=81, value=15, key="m1_meds")
        number_diagnoses   = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=7, key="m1_diag")
        number_emergency   = st.number_input("Emergency Visits (Prior Year)", min_value=0, max_value=76, value=0, key="m1_emerg")

    with col3:
        number_inpatient  = st.number_input("Inpatient Visits (Prior Year)", min_value=0, max_value=21, value=0, key="m1_inp")
        number_outpatient = st.number_input("Outpatient Visits (Prior Year)", min_value=0, max_value=42, value=0, key="m1_outp")

        a1c_options = {"Not Tested": -1, "Normal": 0, "Elevated (>7%)": 1, "High (>8%)": 2}
        a1c_label = st.selectbox("A1C Test Result", list(a1c_options.keys()), key="m1_a1c")
        A1Cresult = a1c_options[a1c_label]

        glu_options = {"Not Tested": -1, "Normal": 0, "High (>200)": 1, "Very High (>300)": 2}
        glu_label = st.selectbox("Max Glucose Serum", list(glu_options.keys()), key="m1_glu")
        max_glu_serum = glu_options[glu_label]

        diabetes_med = st.selectbox("On Diabetes Medication?", ["Yes", "No"], key="m1_diabmed")
        diabetes_med_encoded = 1 if diabetes_med == "Yes" else 0

        insulin = st.selectbox("Insulin?", ["No", "Steady", "Up", "Down"], key="m1_insulin")
        on_insulin = 0 if insulin == "No" else 1

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        diag_1 = st.selectbox("Primary Diagnosis", ["circulatory","diabetes","digestive","external","missing","other","respiratory"], key="m1_d1")
        admission_type = st.selectbox("Admission Type", [1,2,3,4,5,6,7,8], key="m1_admtype")

    with col_b:
        diag_2 = st.selectbox("Secondary Diagnosis", ["circulatory","diabetes","digestive","external","missing","other","respiratory"], key="m1_d2")
        admission_source = st.selectbox("Admission Source", [1,2,3,4,5,6,7,8,9,10,11,13,14,17,20,22,25], key="m1_admsrc")

    with col_c:
        diag_3 = st.selectbox("Third Diagnosis", ["circulatory","diabetes","digestive","external","missing","other","respiratory"], key="m1_d3")
        discharge_disposition = st.selectbox("Discharge Disposition", [1,2,3,4,5,6,7,8,9,10,12,15,16,17,18,22,23,24,25,27,28], key="m1_disc")

    if st.button("Predict Readmission Risk", key="m1_predict"):
        # build base input row — all zeros first
        input_df = pd.DataFrame([{col: 0 for col in feature_cols1}])

        # fill in numeric features
        input_df['age']                = age
        input_df['time_in_hospital']   = time_in_hospital
        input_df['num_lab_procedures'] = num_lab_procedures
        input_df['num_procedures']     = num_procedures
        input_df['num_medications']    = num_medications
        input_df['number_diagnoses']   = number_diagnoses
        input_df['number_emergency']   = number_emergency
        input_df['number_inpatient']   = number_inpatient
        input_df['max_glu_serum']      = max_glu_serum
        input_df['A1Cresult']          = A1Cresult
        input_df['gender']             = gender_encoded
        input_df['weight_checked']     = weight_checked
        input_df['diabetesMed']        = diabetes_med_encoded
        input_df['on_insulin']         = on_insulin

        # engineered features — must match pipeline
        input_df['total_prior_visits']    = number_inpatient + number_outpatient + number_emergency
        input_df['high_risk_prior']       = 1 if number_inpatient >= 2 else 0
        input_df['los_x_inpatient']       = time_in_hospital * number_inpatient
        input_df['inpatient_x_medications'] = number_inpatient * num_medications
        input_df['complexity_score']      = num_lab_procedures + num_medications + number_diagnoses + number_inpatient + number_outpatient + number_emergency

        # one-hot encoded fields — set the matching column to 1 if it exists
        for col_name in [
            f"race_{race}",
            f"diag_1_{diag_1}",
            f"diag_2_{diag_2}",
            f"diag_3_{diag_3}",
            f"admission_type_id_{admission_type}",
            f"admission_source_id_{admission_source}",
            f"discharge_disposition_id_{discharge_disposition}",
        ]:
            if col_name in input_df.columns:
                input_df[col_name] = 1

        # align to training column order
        input_df = input_df[feature_cols1]

        proba = model1.predict_proba(input_df)[0][1]
        pred  = int(proba >= 0.5)
        confidence = abs(proba - 0.5) * 2

        st.markdown("---")
        if pred == 1:
            st.error(f"⚠️ High Readmission Risk — Probability: {proba:.1%}")
        else:
            st.success(f"✅ Low Readmission Risk — Probability: {proba:.1%}")

        st.write(f"**Model Confidence:** {confidence:.1%}")
        st.caption("This prediction is intended to support clinical decision-making, not replace it.")
#model 2 buffer -----
elif model_choice == "Model 2: Deep Learning":
    st.header("Model 2: Readmission Risk — Deep Neural Network")
    st.write("Same readmission prediction task as Model 1, using a Keras DNN instead of XGBoost.")

    @st.cache_resource
    def load_model2():
        import tensorflow as tf

        model_url   = "https://huggingface.co/jfranklingoff/Capstone-Project-Deep-Learning/resolve/main/model.keras"
        cols_url    = "https://huggingface.co/jfranklingoff/Capstone-Project-Deep-Learning/resolve/main/feature_cols.joblib"
        imputer_url = "https://huggingface.co/jfranklingoff/Capstone-Project-Deep-Learning/resolve/main/imputer.joblib"
        scaler_url  = "https://huggingface.co/jfranklingoff/Capstone-Project-Deep-Learning/resolve/main/scaler.joblib"

        model_path   = Path("model2_model.keras")
        cols_path    = Path("model2_feature_cols.joblib")
        imputer_path = Path("model2_imputer.joblib")
        scaler_path  = Path("model2_scaler.joblib")

        for url, path in [
            (model_url,   model_path),
            (cols_url,    cols_path),
            (imputer_url, imputer_path),
            (scaler_url,  scaler_path),
        ]:
            if not path.exists():
                r = requests.get(url)
                r.raise_for_status()
                path.write_bytes(r.content)

        model        = tf.keras.models.load_model(model_path)
        feature_cols = joblib.load(cols_path)
        imputer      = joblib.load(imputer_path)
        scaler       = joblib.load(scaler_path)
        return model, feature_cols, imputer, scaler

    with st.spinner("Loading Model 2 (this may take a moment)..."):
        model2, feature_cols2, imputer2, scaler2 = load_model2()

    st.subheader("Enter Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_options2 = {
            "0–10": 5, "10–20": 15, "20–30": 25, "30–40": 35, "40–50": 45,
            "50–60": 55, "60–70": 65, "70–80": 75, "80–90": 85, "90–100": 95
        }
        age_label2 = st.selectbox("Age Range", list(age_options2.keys()), key="m2_age")
        age2 = age_options2[age_label2]

        gender2 = st.selectbox("Gender", ["Male", "Female"], key="m2_gender")
        gender_encoded2 = 0 if gender2 == "Male" else 1

        race2 = st.selectbox("Race", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic"], key="m2_race")

        time_in_hospital2 = st.number_input("Days in Hospital", min_value=1, max_value=14, value=4, key="m2_los")

        weight_checked2 = st.selectbox("Weight Recorded?", ["No", "Yes"], key="m2_weight")
        weight_checked2 = 1 if weight_checked2 == "Yes" else 0

    with col2:
        num_lab_procedures2 = st.number_input("Number of Lab Procedures", min_value=1, max_value=132, value=40, key="m2_labs")
        num_procedures2     = st.number_input("Number of Medical Procedures", min_value=0, max_value=6, value=1, key="m2_procs")
        num_medications2    = st.number_input("Number of Medications", min_value=1, max_value=81, value=15, key="m2_meds")
        number_diagnoses2   = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=7, key="m2_diag")
        number_emergency2   = st.number_input("Emergency Visits (Prior Year)", min_value=0, max_value=76, value=0, key="m2_emerg")

    with col3:
        number_inpatient2  = st.number_input("Inpatient Visits (Prior Year)", min_value=0, max_value=21, value=0, key="m2_inp")
        number_outpatient2 = st.number_input("Outpatient Visits (Prior Year)", min_value=0, max_value=42, value=0, key="m2_outp")

        a1c_options2 = {"Not Tested": -1, "Normal": 0, "Elevated (>7%)": 1, "High (>8%)": 2}
        a1c_label2 = st.selectbox("A1C Test Result", list(a1c_options2.keys()), key="m2_a1c")
        A1Cresult2 = a1c_options2[a1c_label2]

        glu_options2 = {"Not Tested": -1, "Normal": 0, "High (>200)": 1, "Very High (>300)": 2}
        glu_label2 = st.selectbox("Max Glucose Serum", list(glu_options2.keys()), key="m2_glu")
        max_glu_serum2 = glu_options2[glu_label2]

        diabetes_med2 = st.selectbox("On Diabetes Medication?", ["Yes", "No"], key="m2_diabmed")
        diabetes_med_encoded2 = 1 if diabetes_med2 == "Yes" else 0

        insulin2 = st.selectbox("Insulin?", ["No", "Steady", "Up", "Down"], key="m2_insulin")
        on_insulin2 = 0 if insulin2 == "No" else 1

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        diag_1b = st.selectbox("Primary Diagnosis", ["circulatory","diabetes","digestive","external","missing","other","respiratory"], key="m2_d1")
        admission_type2 = st.selectbox("Admission Type", [1,2,3,4,5,6,7,8], key="m2_admtype")

    with col_b:
        diag_2b = st.selectbox("Secondary Diagnosis", ["circulatory","diabetes","digestive","external","missing","other","respiratory"], key="m2_d2")
        admission_source2 = st.selectbox("Admission Source", [1,2,3,4,5,6,7,8,9,10,11,13,14,17,20,22,25], key="m2_admsrc")

    with col_c:
        diag_3b = st.selectbox("Third Diagnosis", ["circulatory","diabetes","digestive","external","missing","other","respiratory"], key="m2_d3")
        discharge_disposition2 = st.selectbox("Discharge Disposition", [1,2,3,4,5,6,7,8,9,10,12,15,16,17,18,22,23,24,25,27,28], key="m2_disc")

    if st.button("Predict Readmission Risk", key="m2_predict"):
        # build base input row — all zeros first
        input_df2 = pd.DataFrame([{col: 0 for col in feature_cols2}])

        # fill numeric features
        input_df2['age']                = age2
        input_df2['time_in_hospital']   = time_in_hospital2
        input_df2['num_lab_procedures'] = num_lab_procedures2
        input_df2['num_procedures']     = num_procedures2
        input_df2['num_medications']    = num_medications2
        input_df2['number_diagnoses']   = number_diagnoses2
        input_df2['number_emergency']   = number_emergency2
        input_df2['number_inpatient']   = number_inpatient2
        input_df2['max_glu_serum']      = max_glu_serum2
        input_df2['A1Cresult']          = A1Cresult2
        input_df2['gender']             = gender_encoded2
        input_df2['weight_checked']     = weight_checked2
        input_df2['diabetesMed']        = diabetes_med_encoded2
        input_df2['on_insulin']         = on_insulin2

        # engineered features — must match pipeline exactly
        input_df2['total_prior_visits']      = number_inpatient2 + number_outpatient2 + number_emergency2
        input_df2['high_risk_prior']         = 1 if number_inpatient2 >= 2 else 0
        input_df2['los_x_inpatient']         = time_in_hospital2 * number_inpatient2
        input_df2['inpatient_x_medications'] = number_inpatient2 * num_medications2
        input_df2['complexity_score']        = num_lab_procedures2 + num_medications2 + number_diagnoses2 + number_inpatient2 + number_outpatient2 + number_emergency2

        # one-hot encoded fields
        for col_name in [
            f"race_{race2}",
            f"diag_1_{diag_1b}",
            f"diag_2_{diag_2b}",
            f"diag_3_{diag_3b}",
            f"admission_type_id_{admission_type2}",
            f"admission_source_id_{admission_source2}",
            f"discharge_disposition_id_{discharge_disposition2}",
        ]:
            if col_name in input_df2.columns:
                input_df2[col_name] = 1

        # align columns then apply saved imputer and scaler
        input_df2 = input_df2[feature_cols2]
        X = imputer2.transform(input_df2)
        X = scaler2.transform(X)

        proba2 = float(model2.predict(X, verbose=0).flatten()[0])
        pred2  = int(proba2 >= 0.5)
        confidence2 = abs(proba2 - 0.5) * 2

        st.markdown("---")
        if pred2 == 1:
            st.error(f"⚠️ High Readmission Risk — Probability: {proba2:.1%}")
        else:
            st.success(f"✅ Low Readmission Risk — Probability: {proba2:.1%}")

        st.write(f"**Model Confidence:** {confidence2:.1%}")
        st.caption("This prediction is intended to support clinical decision-making, not replace it.")
#model3 buffer ----- 
elif model_choice == "Model 3: CNN (Image Classification)":
    st.header("Model 3: CNN — Image Classification")

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
    # @st.cache_resource
    # def load_model3():
    #     import tensorflow as tf
    #     return tf.keras.models.load_model("models/model3_cnn/saved_model/model.keras")
    #
    # model = load_model3()
    #
    # uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    # if uploaded_file is not None:
    #     from PIL import Image
    #     import numpy as np
    #
    #     image = Image.open(uploaded_file)
    #     st.image(image, caption="Uploaded Image", use_container_width=True)
    #
    #     # Preprocess — must match your training preprocessing
    #     img_resized = image.resize((224, 224))
    #     img_array = np.array(img_resized) / 255.0
    #     img_batch = np.expand_dims(img_array, axis=0)
    #
    #     if st.button("Classify"):
    #         prediction = model.predict(img_batch)
    #         confidence = float(prediction.max())
    #         predicted_class = "Positive" if prediction[0][0] > 0.5 else "Negative"
    #         st.success(f"Prediction: {predicted_class}")
    #         st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    # ---- INTEGRATION PATTERN (uncomment and adapt) ----
  
    @st.cache_resource
    def load_model3():
        import tensorflow as tf

        model_url = "https://huggingface.co/jfranklingoff/Capstone-Project-CNN-Model/resolve/main/cnn_model.keras"

        local_model_path = tf.keras.utils.get_file(
            "cnn_model.keras",
            origin=model_url
        )

        return tf.keras.models.load_model(local_model_path)
    
    model = load_model3()
    
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        from PIL import Image
        import numpy as np
    
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
        # Preprocess — must match your training preprocessing
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
    
        if st.button("Classify"):
            prediction = model.predict(img_batch)
            confidence = float(prediction.max())
            predicted_class = "Positive" if prediction[0][0] > 0.5 else "Negative"
            st.success(f"Prediction: {predicted_class}")
            st.write(f"Confidence: {confidence:.2%}")
    # ---- END PATTERN ----

    st.info("""
            Add a retinal scan image file by dragging and dropping a file, or select a file using the Browse files button.

            Once uploaded, your selected image will be displayed. Click Classify to predict a classification with confidence score.
            """
            )

elif model_choice == "Model 4: NLP (Text Classification)":
    st.header("Model 4: NLP — Text Classification")

    import re

    # Must match the cleaning used in models/model4_nlp_classification/predict.py
    CONTRACTIONS = [
        (r"won't", "will not"),
        (r"can't", "can not"),
        (r"shan't", "shall not"),
        (r"n't", " not"),
        (r"'re", " are"),
        (r"'ve", " have"),
        (r"'ll", " will"),
        (r"'d", " would"),
        (r"'m", " am"),
        (r"it's", "it is"),
        (r"that's", "that is"),
        (r"what's", "what is"),
        (r"there's", "there is"),
    ]

    def clean_text(text):
        if not isinstance(text, str) or text.strip() == '':
            return ''
        text = text.lower()
        for pat, repl in CONTRACTIONS:
            text = re.sub(pat, repl, text)
        text = re.sub(r'<.*?>', ' ', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'%u[0-9a-fA-F]{4}', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @st.cache_resource
    def load_model4():
        
        #for local load
        #model_dir = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
        #model = joblib.load(model_dir / "model.joblib")
        #vectorizer = joblib.load(model_dir / "tfidf_vectorizer.joblib")
        #return model, vectorizer

        model_url = "https://huggingface.co/jfranklingoff/Capstone-Project-NLP-Models/resolve/main/model4_nlp_model.joblib"
        vectorizer_url = "https://huggingface.co/jfranklingoff/Capstone-Project-NLP-Models/resolve/main/model4_tfidf_vectorizer.joblib"

        model_path = Path("model4_nlp_model.joblib")
        vectorizer_path = Path("model4_tfidf_vectorizer.joblib")

        # Download model if missing
        if not model_path.exists():
            response = requests.get(model_url)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                f.write(response.content)

        # Download vectorizer if missing
        if not vectorizer_path.exists():
            response = requests.get(vectorizer_url)
            response.raise_for_status()

            with open(vectorizer_path, "wb") as f:
                f.write(response.content)

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        return model, vectorizer

    model, vectorizer = load_model4()

    user_text = st.text_area("Enter patient medication feedback to classify:", height=150)
    if st.button("Classify") and user_text:
        cleaned = clean_text(user_text)
        text_vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(text_vectorized)[0]
        confidence = model.predict_proba(text_vectorized).max()
        st.success(f"Predicted Effectiveness: {prediction}")
        st.write(f"Confidence: {confidence:.2%}")

elif model_choice == "Model 5: XGBoost":
    st.header("Model 5: XGBoost Diabetes Medication Predictor")
    # TODO: Add your custom model interface
    st.info("This model predicts whether a patient is likely to need diabetes medication based on non-medication clinical and demographic information.")
    
    @st.cache_resource
    def load_model5():
        
        #for local/github loading
        #PROJECT_ROOT = Path(__file__).resolve().parents[1]
        #model_path = PROJECT_ROOT / "models" / "model5_innovation" / "saved_model" / "model.joblib"
        #return joblib.load(model_path)

        model_url = "https://huggingface.co/jfranklingoff/Capstone-Project-XGBoost/resolve/main/model5_xgboost_model.joblib"
        local_model_path = "model5_xgboost_model.joblib"

        urlretrieve(model_url, local_model_path)

        return joblib.load(local_model_path)


    #load model 5
    model = load_model5()

    st.subheader("Enter Patient Information")

    #creates 3 columns for cleaner ui
    col1, col2, col3 = st.columns(3)

    #col 1
    with col1:

        # Convert age range selections into numeric midpoint values
        # Matches preprocessing logic used during training
        age_options = {
            "0–10": 5,
            "10–20": 15,
            "20–30": 25,
            "30–40": 35,
            "40–50": 45,
            "50–60": 55,
            "60–70": 65,
            "70–80": 75,
            "80–90": 85,
            "90–100": 95
        }

        # User selects age range
        age_label = st.selectbox(
            "Age Range",
            list(age_options.keys())
        )

        # Convert selected age range into numeric value
        age = age_options[age_label]

        # Binary encode whether weight was recorded
        weight_checked = st.selectbox(
            "Weight Recorded?",
            ["No", "Yes"]
        )
        weight_checked = 1 if weight_checked == "Yes" else 0

        # Numeric hospital visit features
        number_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=76, value=38)

        number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, max_value=16, value=8)

        #Primary diagnosis category selection
        diag_1 = st.selectbox(
            "Primary Diagnosis Category",
            [
                "circulatory",
                "diabetes",
                "digestive",
                "external",
                "missing",
                "other",
                "respiratory"
            ]
        )

        #Admission source category
        admission_source = st.selectbox(
            "Admission Source",
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 20, 22, 25]
        )

        # Glucose result options mapped to ordinal encoding
        # Matches training pipeline encoding
        max_glu_options = {
            "Not Tested": -1,
            "Normal": 0,
            "High (>200 mg/dL)": 1,
            "Very High (>300 mg/dL)": 2
        }

        max_glu_label = st.selectbox(
            "Maximum Glucose Serum Result",
            list(max_glu_options.keys())
        )

        max_glu_serum = max_glu_options[max_glu_label]


    with col2:
        
        #Binary encode gender to match training data
        gender = st.selectbox("Gender", ["Male", "Female"])

        gender_encoded = 0 if gender == "Male" else 1

        # Binary encode previous readmission history
        readmission_binary = st.selectbox(
            "Readmitted Previously?",
            ["No", "Yes"]
        )

        readmission_binary = 1 if readmission_binary == "Yes" else 0

        # Numeric clinical features
        number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=21, value=11)
        
        num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=132, value=65)

        # Secondary diagnosis category
        diag_2 = st.selectbox(
            "Secondary Diagnosis Category",
            [
                "circulatory",
                "diabetes",
                "digestive",
                "external",
                "missing",
                "other",
                "respiratory"
            ]
        )

        # Admission type category
        admission_type = st.selectbox(
            "Admission Type",
            [2, 3, 4, 5, 6, 7, 8]
        )



    with col3:

        # Race category (used for one-hot encoding)
        race = st.selectbox(
            "Race",
            [
                "AfricanAmerican",
                "Asian",
                "Caucasian",
                "Hispanic"
            ]
        )

        # Hospital stay length
        time_in_hospital = st.number_input("Days in Hospital", min_value=1, max_value=14, value=7)

        # Outpatient visit history
        number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=42, value=21)

        # Number of procedures performed
        num_procedures = st.number_input("Number of Medical Procedures", min_value=0, max_value=6, value=3)
        
        # Third diagnosis category
        diag_3 = st.selectbox(
            "Third Diagnosis Category",
            [
                "circulatory",
                "diabetes",
                "digestive",
                "external",
                "missing",
                "other",
                "respiratory"
            ]
        )

        # Discharge disposition category
        discharge_disposition = st.selectbox(
            "Discharge Disposition",
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 18, 22, 23, 24, 25, 27, 28]
        )

        # A1C result mapped to ordinal encoding
        a1c_options = {
            "Not Tested": -1,
            "Normal": 0,
            "Elevated (>7%)": 1,
            "High (>8%)": 2
        }

        a1c_label = st.selectbox(
            "A1C Test Result",
            list(a1c_options.keys())
        )

        A1Cresult = a1c_options[a1c_label]



    # Create single-row DataFrame from user inputs
    # This matches the model input format
    input_df = pd.DataFrame([{
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "gender": gender_encoded,
        "weight_checked": weight_checked,
        "readmission_binary": readmission_binary,
    }])

    #ensures non selected items in select boxes are set to 0
    for col in model.feature_names_in_:
        if col not in input_df.columns:
            input_df[col] = 0

    # One-hot encode selected race value
    race_col = f"race_{race}"

    if race_col in input_df.columns:
        input_df[race_col] = 1

    # One-hot encode diagnosis categories
    diag_1_col = f"diag_1_{diag_1}"

    if diag_1_col in input_df.columns:
        input_df[diag_1_col] = 1
    
    diag_2_col = f"diag_2_{diag_2}"

    if diag_2_col in input_df.columns:
        input_df[diag_2_col] = 1

    diag_3_col = f"diag_3_{diag_3}"

    if diag_3_col in input_df.columns:
        input_df[diag_3_col] = 1

    # One-hot encode admission type
    admission_type_col = f"admission_type_id_{admission_type}"

    if admission_type_col in input_df.columns:
        input_df[admission_type_col] = 1

    # One-hot encode discharge disposition
    discharge_col = f"discharge_disposition_id_{discharge_disposition}"

    if discharge_col in input_df.columns:
        input_df[discharge_col] = 1

    # One-hot encode admission source
    admission_source_col = f"admission_source_id_{admission_source}"

    if admission_source_col in input_df.columns:
        input_df[admission_source_col] = 1

    # Reorder columns to match expected input order
    input_df = input_df[list(model.feature_names_in_)]

    # Run prediction when button is clicked
    if st.button("Predict Diabetes Medication Need"):
        
        # Get probability for positive class (needs medication)
        confidence = model.predict_proba(input_df)[:, 1][0]

        # Use tuned threshold from model evaluation
        # Default XGBoost threshold is 0.50, but 0.70 gave better class balance
        threshold = 0.70

        # Convert probability into final prediction
        prediction =1 if confidence >= threshold else 0

        # Display prediction result to user
        if prediction == 1:
            st.success("Prediction: Patient is likely to need diabetes medication.")
        else:
            st.warning("Prediction: Patient is not likely to need diabetes medication.")

        # Display model confidence as percentage
        st.write(f"confidence: {confidence:.2%}")



elif model_choice == "Our Team":
    st.header("Our Team")
    st.write("Please meet the members of Aegis Health Strategy")
    st.info("Clifton Rand: Cliff is our Data Engineer and Model 5 lead")
    st.info("Jesse Goff: Jesse is our Model 3 and Presentation lead")
    st.info("Sean McManus: Sean is our Model 1 lead")
    st.info("Brodie Ellis: Brodie is our Model 4 lead")