import pandas as pd
import joblib

MODEL_PATH = "model/churn_model.pkl"
COLUMNS_PATH = "model/model_columns.pkl"
SCALER_PATH = "model/scaler.pkl"   # agar naam alag hai to yahan theek kar do
DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

X = df.drop(columns=["customerID", "Churn"])
X = pd.get_dummies(X)
X = X.reindex(columns=columns, fill_value=0)

# Scaler apply karo (khud detect karta hai kin columns par)
if hasattr(scaler, "feature_names_in_"):
    cols_to_scale = list(scaler.feature_names_in_)
    X[cols_to_scale] = scaler.transform(X[cols_to_scale])
elif scaler.n_features_in_ == X.shape[1]:
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)
else:
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    X[num_cols] = scaler.transform(X[num_cols])

df["Churn_Probability"] = model.predict_proba(X)[:, 1]
df["Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Low", "Medium", "High"],
    include_lowest=True,
)

out = df[["customerID", "Churn_Probability", "Risk_Level"]]
out.to_csv("predictions.csv", index=False)

print(out.head())
print("Total rows:", len(out))
print(out["Risk_Level"].value_counts())