from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import numpy as np

app = FastAPI(title="Churn Prediction API")

model = joblib.load("../model/churn_model.pkl")
scaler = joblib.load("../model/scaler.pkl")
model_columns = joblib.load("../model/model_columns.pkl")

# Create a SHAP explainer once, at startup (using zeros as a simple background reference)
background = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)
explainer = shap.LinearExplainer(model, background)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def preprocess(customer: Customer):
    input_df = pd.DataFrame([customer.dict()])
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
    return input_df

@app.post("/predict")
def predict(customer: Customer):
    input_df = preprocess(customer)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # SHAP values for this specific customer
    shap_values = explainer.shap_values(input_df)[0]

    # Get top 5 features pushing toward/away from churn
    contributions = list(zip(model_columns, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_factors = [
        {"feature": feat, "impact": round(float(val), 4)}
        for feat, val in contributions[:5]
    ]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(probability), 4),
        "top_factors": top_factors
    }