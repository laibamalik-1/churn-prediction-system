import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

st.set_page_config(page_title="Churn Prediction Dashboard", layout="centered")

# Load model, scaler, columns (cached so it only loads once)
@st.cache_resource
def load_artifacts():
    model = joblib.load("model/churn_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    model_columns = joblib.load("model/model_columns.pkl")
    background = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
    explainer = shap.LinearExplainer(model, background)
    return model, scaler, model_columns, explainer

model, scaler, model_columns, explainer = load_artifacts()

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter customer details to predict churn risk.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with col2:
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=800.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    raw_input = {
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    input_df = pd.DataFrame([raw_input])
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    multi_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=multi_cat_cols)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_columns]
    input_df = input_df.astype(float)

    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Churn Risk — Probability: {probability*100:.2f}%")
    else:
        st.success(f"✅ Low Churn Risk — Probability: {probability*100:.2f}%")

    shap_values = explainer.shap_values(input_df)[0]
    contributions = list(zip(model_columns, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    st.subheader("Why this prediction?")
    st.write("Top factors influencing this result:")
    for feat, val in contributions[:5]:
        direction = "🔺 Increases churn risk" if val > 0 else "🔻 Decreases churn risk"
        st.write(f"**{feat}** — {direction} (impact: {val:.3f})")