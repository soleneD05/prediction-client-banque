from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from src.utils import load_params

params = load_params()
model_path = params["model"]["path"]
preprocessor_path = params["preprocessing"]["preprocessor_path"]

# Charger le modèle
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Charger les composants de preprocessing (plus simple!)
with open(preprocessor_path, "rb") as f:
    preprocessor_data = pickle.load(f)

encoders = preprocessor_data['encoders']
scaler = preprocessor_data['scaler']
categorical_columns = preprocessor_data['categorical_columns']
numerical_columns = preprocessor_data['numerical_columns']

print("✅ Model and preprocessor loaded successfully!")

class CustomerData(BaseModel):
    Surname: str
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

app = FastAPI(
    title="Prédiction de Churn",
    description="Application de prédiction de Churn"
)

def preprocess_data(df):
    """Transform new data for prediction"""
    df_processed = df.copy()
    
    # Apply label encoding to categorical columns
    for col in categorical_columns:
        if col in df_processed.columns and col in encoders:
            le = encoders[col]
            # Handle unknown categories
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df_processed[col] = le.transform(df_processed[col])
    
    # Apply scaling to numerical columns
    if numerical_columns:
        df_processed[numerical_columns] = scaler.transform(df_processed[numerical_columns])
    
    return df_processed

@app.post("/predict", tags=["Predict"])
async def predict(data: CustomerData) -> str:
    """Prédit si un client va churner""" 
    df = pd.DataFrame([data.model_dump()])
    preprocessed_data = preprocess_data(df)
    prediction = model.predict(preprocessed_data)[0]
    return "Exited" if prediction == 1 else "Not Exited"

@app.get("/")
async def root():
    return {"message": "API de prédiction de churn - opérationnelle!"}