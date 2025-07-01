import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump
import os

def train_model(processed_data_path, model_save_path):
    """Trains a model on the processed data."""
    
    # Load the processed data
    df = pd.read_csv(processed_data_path)
    
    # Identify non-numeric or irrelevant columns to drop from features
    columns_to_drop = ['transaction_date', 'customer_id', 'transaction_id', 'target'] 
    
    # Identify categorical columns that need to be one-hot encoded
    categorical_cols = ['gender', 'education', 'marital_status']
    
    # Drop irrelevant columns first
    X = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Perform one-hot encoding on specified categorical columns
    X = pd.get_dummies(X, columns=categorical_cols)  # Removed drop_first=True
    
    # The 'is_high_risk' column is the target variable
    y = X['is_high_risk']
    X = X.drop(columns=['is_high_risk']) # Remove target from features
    
    # Train a simple model
    model = LogisticRegression(solver='liblinear', random_state=42) 
    model.fit(X, y)
    
    # Save the model
    dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    processed_data_path = 'data/processed/processed_credit_risk_data.csv'
    model_save_path = 'src/model.joblib'
    
    # Check if processed data exists
    if not os.path.exists(processed_data_path):
        print("Processed data not found. Please run src/data_processing.py first.")
    else:
        train_model(processed_data_path, model_save_path)