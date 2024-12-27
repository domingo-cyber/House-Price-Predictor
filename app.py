from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)


        
        # Rest of your prediction code...

# Check if model exists, if not, train it
if not os.path.exists('models/model.pkl') or not os.path.exists('models/scaler.pkl'):
    print("Model or scaler not found. Training new model...")
    import download_data
    import model_training
    download_data.create_sample_housing_data()
    model_training.__main__

# Load the model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        import time
        time.sleep(1)
        # Get values from the form
        features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                   'waterfront', 'view', 'condition', 'grade', 'yr_built']
        
        input_values = []
        for feature in features:
            value = float(request.form[feature])
            input_values.append(value)
        
        # Convert to numpy array and reshape
        input_array = np.array(input_values).reshape(1, -1)
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        return render_template('result.html', 
                             prediction=f"${prediction:,.2f}")
    
    except Exception as e:
        return render_template('result.html', 
                             error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)