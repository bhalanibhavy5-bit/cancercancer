import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")

st.title("🩺 Breast Cancer Diagnosis Prediction App")
st.write("""
This app uses a Support Vector Machine (SVC) classifier trained on Principal Components 
to predict whether a breast mass is **Malignant (M)** or **Benign (B)**.
""")

# Load saved models and preprocessing pipelines safely
@st.cache_resource
def load_assets():
    scaler = joblib.load('models/scaler.joblib')
    pca = joblib.load('models/pca_transformer.joblib')
    model = joblib.load('models/svc_model.joblib')
    return scaler, pca, model

try:
    scaler, pca, model = load_assets()
except Exception as e:
    st.error(f"Error loading models: {e}. Please make sure you have executed the previous cells to save the models in the 'models/' folder.")
    st.stop()

# Feature names matching the original dataset
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

st.sidebar.header("🔬 Input Patient Clinical Metrics")

# Group inputs dynamically into columns for a better UI layout
input_data = {}
cols = st.columns(3)
for idx, col in enumerate(feature_names):
    col_target = cols[idx % 3]
    # Get a reasonable default based on 0 (since data will be scaled)
    val = col_target.number_input(f"{col.replace('_', ' ').title()}", value=0.0, format="%.4f")
    input_data[col] = val

if st.button("Run Diagnostic Prediction", type="primary"):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess inputs
    input_scaled = scaler.transform(input_df)
    input_pca = pca.transform(input_scaled)
    
    # Predict
    prediction = model.predict(input_pca)[0]
    probabilities = getattr(model, "predict_proba", None)
    
    st.subheader("Prediction Results")
    if prediction == 'M':
        st.error("🚨 **Diagnosis Prediction: Malignant (M)**")
        st.write("The model suggests a high likelihood that the tumor is malignant.")
    else:
        st.success("🟢 **Diagnosis Prediction: Benign (B)**")
        st.write("The model suggests a high likelihood that the tumor is benign.")
