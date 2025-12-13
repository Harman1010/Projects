# Projects
📞 Telco Customer Churn Prediction
A machine learning project that predicts whether a telecom customer will churn and explains why using SHAP values. The model is deployed using a simple Gradio interface.

🚀 Features
XGBoost model trained with class balancing and hyperparameter tuning
Threshold tuning (0.45) to improve churn recall
Local SHAP explainability for each user:
SHAP summary plot
SHAP force plot

🧠 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Gradio, Hugging Face Spaces

🧪 Model Performance
After tuning:

Precision (Churn): ~0.54
Recall (Churn): ~0.71
F1 Score (Churn): ~0.61
Optimized to detect customers at high risk of churn.

🖥 Deployment
The app takes customer inputs → predicts churn probability

Files included:

app.py (Gradio app)
requirements.txt
model.pkl (trained model)
columns.json (training feature order)
requirements.txt
