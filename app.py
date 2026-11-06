import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    layout="wide"
)

st.title("🩺 Breast Cancer Diagnosis Prediction App")

st.markdown("""
This application predicts whether a breast tumor is:

- **Malignant (M)**
- **Benign (B)**

using a Support Vector Machine (SVM) model trained on the Wisconsin Breast Cancer dataset.
""")

# ----------------------------------------------------
# LOAD DATA & TRAIN MODEL
# ----------------------------------------------------
@st.cache_resource
def train_model():

    df = pd.read_csv("breast-cancer-wisconsin-data.csv")

    # Remove unnamed column if present
    if "Unnamed: 32" in df.columns:
        df = df.drop("Unnamed: 32", axis=1)

    # Remove ID column if present
    if "id" in df.columns:
        df = df.drop("id", axis=1)

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_pca, y)

    return scaler, pca, model, X


try:
    scaler, pca, model, X_ref = train_model()

except Exception as e:
    st.error(f"Dataset loading failed: {e}")
    st.stop()

# ----------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------
st.sidebar.header("📋 Input Clinical Features")

user_input = {}

cols = st.columns(3)

for i, feature in enumerate(X_ref.columns):

    col = cols[i % 3]

    default_value = float(X_ref[feature].median())

    value = col.number_input(
        feature,
        value=default_value,
        format="%.4f"
    )

    user_input[feature] = value

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if st.button("🔍 Run Prediction"):

    input_df = pd.DataFrame([user_input])

    input_scaled = scaler.transform(input_df)

    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)[0]

    probability = model.predict_proba(input_pca)

    st.subheader("Prediction Result")

    if prediction == "M":

        st.error("🚨 Malignant (M)")

        st.metric(
            "Probability",
            f"{probability.max()*100:.2f}%"
        )

        st.write(
            "The model predicts that the tumor is likely malignant."
        )

    else:

        st.success("✅ Benign (B)")

        st.metric(
            "Probability",
            f"{probability.max()*100:.2f}%"
        )

        st.write(
            "The model predicts that the tumor is likely benign."
        )

# ----------------------------------------------------
# DATA PREVIEW
# ----------------------------------------------------
with st.expander("View Dataset"):

    df = pd.read_csv("breast-cancer-wisconsin-data.csv")

    st.dataframe(df.head())
