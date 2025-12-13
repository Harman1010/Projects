import gradio as gr
import pandas as pd
import numpy as np
import joblib, json
import shap
import matplotlib.pyplot as plt



model = joblib.load("model.pkl")

with open("columns.json", "r") as f:
    MODEL_COLUMNS = json.load(f)

#booster = model.get_booster()
#explainer = shap.TreeExplainer(booster)



# ------------------ PREPROCESS FUNCTION ------------------
def preprocess_input(gender, senior, partner, dependents, tenure,
                     phone, multiple, internet, online_sec, online_backup,
                     device_protect, tech_support, stream_tv, stream_movie,
                     contract, paperless, payment, monthly, total):

    df = pd.DataFrame([[gender, senior, partner, dependents, tenure,
                        phone, multiple, internet, online_sec, online_backup,
                        device_protect, tech_support, stream_tv, stream_movie,
                        contract, paperless, payment, monthly, total]],
                      columns=['gender','SeniorCitizen','Partner','Dependents','tenure',
                               'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
                               'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
                               'Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])

    # Convert to one-hot format
    df = pd.get_dummies(df, drop_first=True)

    # Add missing columns
    for col in MODEL_COLUMNS:
        if col not in df:
            df[col] = 0

    # Reorder to match training layout
    df = df[MODEL_COLUMNS]
    return df


THRESHOLD = 0.45


# ------------------ PREDICTION + SHAP FUNCTION ------------------
def predict_and_shap(gender, senior, partner, dependents, tenure,
                     phone, multiple, internet, online_sec, online_backup,
                     device_protect, tech_support, stream_tv, stream_movie,
                     contract, paperless, payment, monthly, total):

    # Preprocess single-row input
    X = preprocess_input(gender, senior, partner, dependents, tenure,
                         phone, multiple, internet, online_sec, online_backup,
                         device_protect, tech_support, stream_tv, stream_movie,
                         contract, paperless, payment, monthly, total)

    # Prediction
    prob = float(model.predict_proba(X)[0][1])
    pred = "Yes (Likely to Churn)" if prob >= THRESHOLD else "No (Not Likely)"

    result = {
        "Churn Probability (%)": round(prob * 100, 2),
        "Prediction": pred
    }

    return result


# ------------------ GRADIO INTERFACE ------------------
inputs = [
    gr.Dropdown(["Male","Female"], label="Gender"),
    gr.Dropdown([0,1], label="Senior Citizen"),
    gr.Dropdown(["Yes","No"], label="Partner"),
    gr.Dropdown(["Yes","No"], label="Dependents"),
    gr.Slider(0,72,step=1,label="Tenure (months)"),
    gr.Dropdown(["Yes","No"], label="Phone Service"),
    gr.Dropdown(["Yes","No"], label="Multiple Lines"),
    gr.Dropdown(["DSL","Fiber optic","No"], label="Internet Service"),
    gr.Dropdown(["Yes","No"], label="Online Security"),
    gr.Dropdown(["Yes","No"], label="Online Backup"),
    gr.Dropdown(["Yes","No"], label="Device Protection"),
    gr.Dropdown(["Yes","No"], label="Tech Support"),
    gr.Dropdown(["Yes","No"], label="Streaming TV"),
    gr.Dropdown(["Yes","No"], label="Streaming Movies"),
    gr.Dropdown(["Month-to-month","One year","Two year"], label="Contract"),
    gr.Dropdown(["Yes","No"], label="Paperless Billing"),
    gr.Dropdown(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], label="Payment Method"),
    gr.Number(label="Monthly Charges"),
    gr.Number(label="Total Charges")
]

outputs = [
    gr.JSON(label="Prediction")
]

gr.Interface(
    fn=predict_and_shap,
    inputs=inputs,
    outputs=outputs,
    title="Telco Churn Prediction with Local SHAP Explainability",
    description="Enter customer details to predict churn and view SHAP explanations."
).launch()
