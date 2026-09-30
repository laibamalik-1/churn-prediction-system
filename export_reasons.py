import pandas as pd
import joblib

model = joblib.load("model/churn_model.pkl")
columns = joblib.load("model/model_columns.pkl")

reasons = pd.DataFrame({
    "Feature": columns,
    "Coefficient": model.coef_[0],
})
reasons["Impact"] = reasons["Coefficient"].apply(
    lambda c: "Increases churn" if c > 0 else "Reduces churn"
)
reasons["Abs_Coefficient"] = reasons["Coefficient"].abs()
reasons = reasons.sort_values("Abs_Coefficient", ascending=False)

reasons.to_csv("reasons.csv", index=False)
print(reasons.head(10))