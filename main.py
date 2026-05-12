from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    gender : str
    SeniorCitizen : int
    partner : str
    dependents : str
    tenure : int
    MonthlyCharges : float
    TotalCharges : float
    phone_service : str
    multiple_lines : str
    internet_service : str
    online_security : str
    online_backup : str
    device_protection : str
    tech_support : str
    streaming_tv : str
    streaming_movies : str
    contract : str
    paperless_billing : str
    payment_method : str


app = FastAPI()

@app.get("/")
def home():
    return {"Message": "Welcome to the Telco Churn Prediction API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([{
        "gender" : request.gender,
        "SeniorCitizen" : request.SeniorCitizen,
        "Partner" : request.partner,
        "Dependents" : request.dependents,
        "tenure" : request.tenure,
        "MonthlyCharges" : request.MonthlyCharges,
        "TotalCharges" : request.TotalCharges,
        "PhoneService" : request.phone_service,
        "MultipleLines" : request.multiple_lines,
        "InternetService" : request.internet_service,
        "OnlineBackup" : request.online_backup,
        "OnlineSecurity" : request.online_security,
        "DeviceProtection" : request.device_protection,
        "TechSupport" : request.tech_support,
        "StreamingTV" : request.streaming_tv,
        "StreamingMovies" : request.streaming_movies,
        "Contract" : request.contract,
        "PaperlessBilling" : request.paperless_billing,
        "PaymentMethod" : request.payment_method
        }])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        result = "Customer is likely to churn"
    else:
        result = "Customer is not likely tochurn"

    return {"Prediction": result}