import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import os

def create_sample_housing_data():
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Load California Housing dataset
    california = fetch_california_housing()
    
    # Create a DataFrame
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['price'] = california.target * 100000  # Convert to actual prices
    
    # Rename and adjust columns to match our model requirements
    df_modified = pd.DataFrame({
        'bedrooms': df['HouseAge'].apply(lambda x: max(1, min(6, int(x/5)))),
        'bathrooms': df['AveRooms'].apply(lambda x: max(1, min(4, x/2))),
        'sqft_living': df['AveRooms'] * 100,
        'sqft_lot': df['AveBedrms'] * 1000,
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], size=len(df)),
        'waterfront': np.random.choice([0, 1], size=len(df), p=[0.9, 0.1]),
        'view': np.random.choice([0, 1, 2, 3, 4], size=len(df)),
        'condition': np.random.choice([1, 2, 3, 4, 5], size=len(df)),
        'grade': np.random.choice(range(1, 14), size=len(df)),
        'yr_built': np.random.randint(1900, 2020, size=len(df)),
        'price': df['price']
    })
    
    # Save to CSV
    df_modified.to_csv('data/housing_data.csv', index=False)
    print("Dataset created and saved to data/housing_data.csv")
    return df_modified

if __name__ == "__main__":
    df = create_sample_housing_data()
    print("\nFirst few rows of the dataset:")
    print(df.head())