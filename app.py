from flask import Flask, render_template, request, jsonify
import joblib  # Use joblib instead of pickle
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

# Load models and preprocessors
def load_models():
    rf_model = joblib.load('models/rf_model.joblib')  # Load model using joblib
    gb_model = joblib.load('models/gb_model.joblib')  # Load model using joblib
    scaler = joblib.load('models/scaler.joblib')      # Load scaler using joblib
    encoders = joblib.load('models/encoders.joblib')  # Load encoders using joblib
    return rf_model, gb_model, scaler, encoders

# Load metrics and feature importance
def load_metadata():
    with open('static/metrics.json', 'r') as f:
        metrics = json.load(f)
    with open('static/feature_importance.json', 'r') as f:
        feature_importance = json.load(f)
    return metrics, feature_importance

# Load models and metadata
rf_model, gb_model, scaler, encoders = load_models()
metrics, feature_importance = load_metadata()

@app.route('/')
def home():
    return render_template('index.html', 
                         metrics=metrics,
                         feature_importance=feature_importance)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from JSON request
        data = request.get_json()  # This line now gets the JSON data from the request
        age = int(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']

        # Validate input data
        if not (18 <= age <= 70):
            return jsonify({'success': False, 'error': 'Age must be between 18 and 70'}), 400
        if sex not in ['male', 'female']:
            return jsonify({'success': False, 'error': 'Sex must be either "male" or "female"'}), 400
        if not (10 <= bmi <= 50):  # Assuming BMI range between 10 and 50
            return jsonify({'success': False, 'error': 'BMI must be between 10 and 50'}), 400
        if smoker not in ['yes', 'no']:
            return jsonify({'success': False, 'error': 'Smoker must be either "yes" or "no"'}), 400

        # Print received data for debugging purposes
        print(data)
        
        # Create DataFrame
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }])

        # Encode categorical variables
        for col, encoder in encoders.items():
            input_df[col] = encoder.transform(input_df[col])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make predictions with both models
        rf_prediction = rf_model.predict(input_scaled)[0]
        gb_prediction = gb_model.predict(input_scaled)[0]

        # Calculate average prediction
        avg_prediction = (rf_prediction + gb_prediction) / 2

        # Get feature contributions (for Random Forest)
        feature_contributions = {}
        for idx, col in enumerate(input_df.columns):
            contribution = rf_model.feature_importances_[idx] * input_scaled[0][idx]
            feature_contributions[col] = float(contribution)

        # Return predictions and contributions
        return jsonify({
            'success': True,
            'rf_prediction': f'${rf_prediction:,.2f}',
            'gb_prediction': f'${gb_prediction:,.2f}',
            'avg_prediction': f'${avg_prediction:,.2f}',
            'feature_contributions': feature_contributions
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
