import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data():
    df = pd.read_csv('data/housing_data.csv')
    return df

def preprocess_data(df):
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Select features
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
               'waterfront', 'view', 'condition', 'grade', 'yr_built']
    
    X = df[features]
    y = df['price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    
    print("Preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)
    
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    print("Saving model...")
    joblib.dump(model, 'models/model.pkl')
    
    # Print model performance
    print(f"Model R2 Score: {model.score(X_test_scaled, y_test)}")
    print("Model training completed!")