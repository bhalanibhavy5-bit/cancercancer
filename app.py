import streamlit as st
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Breast Cancer Diagnosis Prediction App")

st.markdown("""
This application predicts whether a breast tumor is:

- **Malignant (M)**
- **Benign (B)**

using a Support Vector Machine (SVM) model.
""")

# -----------------------------------
# LOAD DATA & TRAIN MODEL
# -----------------------------------
@st.cache_resource
def train_model():

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = pd.Series(data.target)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_pca,
        y,
        test_size=0.2,
        random_state=42
    )

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    return model, scaler, pca, X, accuracy


model, scaler, pca, X_ref, accuracy = train_model()

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.header("Model Information")
st.sidebar.success(f"Accuracy: {accuracy:.2%}")

st.sidebar.header("Patient Measurements")

# -----------------------------------
# USER INPUTS
# -----------------------------------
user_data = {}

col1, col2, col3 = st.columns(3)

for i, feature in enumerate(X_ref.columns):

    median_value = float(X_ref[feature].median())

    if i % 3 == 0:
        user_data[feature] = col1.number_input(
            feature,
            value=median_value,
            format="%.4f"
        )

    elif i % 3 == 1:
        user_data[feature] = col2.number_input(
            feature,
            value=median_value,
            format="%.4f"
        )

    else:
        user_data[feature] = col3.number_input(
            feature,
            value=median_value,
            format="%.4f"
        )

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🔍 Predict Diagnosis"):

    input_df = pd.DataFrame([user_data])

    input_scaled = scaler.transform(input_df)

    input_pca = pca.transform(input_scaled)

    prediction = model.predict(input_pca)[0]

    probability = model.predict_proba(input_pca)[0]

    malignant_prob = probability[0] * 100
    benign_prob = probability[1] * 100

    st.subheader("Prediction Result")

    if prediction == 0:

        st.error(
            f"🚨 Diagnosis Prediction: MALIGNANT\n\nConfidence: {malignant_prob:.2f}%"
        )

    else:

        st.success(
            f"✅ Diagnosis Prediction: BENIGN\n\nConfidence: {benign_prob:.2f}%"
        )

    st.subheader("Probability Distribution")

    result_df = pd.DataFrame({
        "Diagnosis": ["Malignant", "Benign"],
        "Probability (%)": [malignant_prob, benign_prob]
    })

    st.bar_chart(
        result_df.set_index("Diagnosis")
    )

# -----------------------------------
# DATASET PREVIEW
# -----------------------------------
with st.expander("View Dataset Sample"):

    st.dataframe(
        X_ref.head(),
        use_container_width=True
    )

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit, Scikit-Learn, PCA and SVM."
)
