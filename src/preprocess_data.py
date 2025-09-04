"""
Data Preprocessing Script
Handles encoding, scaling and train/test split for Churn Modelling dataset
Creates preprocessor.pkl for the API
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.utils import load_params
import pickle  # ✅ TOUT avec pickle !

class APIPreprocessor:
    """Preprocessor class for the API - combines encoding and scaling"""
    
    def __init__(self, encoders, scaler, categorical_columns, numerical_columns):
        self.encoders = encoders
        self.scaler = scaler
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        
    def transform(self, X):
        """Transform new data for prediction"""
        X_processed = X.copy()
        
        # Apply label encoding to categorical columns
        for col in self.categorical_columns:
            if col in X_processed.columns and col in self.encoders:
                le = self.encoders[col]
                # Handle unknown categories
                X_processed[col] = X_processed[col].apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X_processed[col] = le.transform(X_processed[col])
        
        # Apply scaling to numerical columns
        if self.numerical_columns:
            X_processed[self.numerical_columns] = self.scaler.transform(X_processed[self.numerical_columns])
        
        return X_processed

def preprocess_data(params: dict):
    """
    What this function does:
    1. Loads the cleaned data
    2. Encodes categorical variables (Geography, Gender)
    3. Separates features (X) and target (y)
    4. Divides into train/test
    5. Normalizes numerical data
    6. Creates and saves preprocessor.pkl for the API
    7. Saves everything for training
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
    
    # ✅ Save encoders avec PICKLE
    with open(params["preprocessing"]["encoders_path"], "wb") as f:
        pickle.dump(encoders, f)
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
    # ATTENTION: Les colonnes Geography et Gender sont maintenant numériques après l'encodage !
    # On ne doit normaliser QUE les vraies colonnes numériques
    true_numerical_columns = [col for col in X_train.columns if col not in categorical_columns]
    print(f"Numerical columns to scale: {true_numerical_columns}")
    
    scaler = StandardScaler()
    X_train[true_numerical_columns] = scaler.fit_transform(X_train[true_numerical_columns])
    X_test[true_numerical_columns] = scaler.transform(X_test[true_numerical_columns]) 
    
    # ✅ Save scaler avec PICKLE
    with open(params["preprocessing"]["scaler_path"], "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved")
    
    # 6. SAVE COMPONENTS FOR THE API (plus simple)
    # Au lieu de créer une classe, sauvegardons les composants séparément
    
    # Save preprocessor components
    preprocessor_data = {
        'encoders': encoders,
        'scaler': scaler,
        'categorical_columns': categorical_columns,
        'numerical_columns': true_numerical_columns
    }
    
    preprocessor_path = params["preprocessing"]["preprocessor_path"]
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor_data, f)
    print(f"✅ preprocessor.pkl saved to: {preprocessor_path}")
    
    # 7. Save everything for training
    X_train.to_csv(params["data"]["X_train_path"], index=False)
    X_test.to_csv(params["data"]["X_test_path"], index=False)
    y_train.to_csv(params["data"]["y_train_path"], index=False)
    y_test.to_csv(params["data"]["y_test_path"], index=False)
    
    print("Preprocessed data saved successfully!")
    print("Ready for model training")

if __name__ == "__main__":
    params = load_params()
    preprocess_data(params)