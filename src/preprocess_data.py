"""
Data Preprocessing Script
Handles encoding, scaling and train/test split for Churn Modelling dataset
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import load_params
import joblib

def preprocess_data(params: dict):
    """
    What this function does:
    1. Loads the cleaned data
    2. Encodes categorical variables (Geography, Gender)
    3. Separates features (X) and target (y)
    4. Divides into train/test
    5. Normalizes numerical data
    6. Saves everything for training
    """
    
    # 1. Load the cleaned data
    clean_data_path = params["data"]["clean_dataset_path"]
    df = pd.read_csv(clean_data_path)
    print(f"Data loaded: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # 2. Encode categorical variables
    categorical_columns = params["preprocessing"]["categorical_columns"]
    encoders = {}
    
    print("Encoding categorical variables...")
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            print(f"  {col}: {df[col].unique()} -> ", end="")
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            print(f"{le.classes_}")
        else:
            print(f"Warning: Column {col} not found in dataset")
    
    # Save encoders
    joblib.dump(encoders, params["preprocessing"]["encoders_path"])
    print("Encoders saved")
    
    # 3. Separate features (X) and target (y)
    target_column = params["data"]["target_column"]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # 4. Divide into train/test
    test_size = params["preprocessing"]["test_size"]
    random_state = params["preprocessing"]["random_state"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y 
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 5. Normalize numerical data
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Numerical columns to scale: {numerical_columns}")
    
    scaler = StandardScaler()
    X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test[numerical_columns] = scaler.transform(X_test[numerical_columns]) 
    
    # Save scaler
    joblib.dump(scaler, params["preprocessing"]["scaler_path"])
    print("Scaler saved")
    
    # 6. Save everything for training
    X_train.to_csv(params["data"]["X_train_path"], index=False)
    X_test.to_csv(params["data"]["X_test_path"], index=False)
    y_train.to_csv(params["data"]["y_train_path"], index=False)
    y_test.to_csv(params["data"]["y_test_path"], index=False)
    
    print("Preprocessed data saved successfully!")
    print("Ready for model training")

if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)