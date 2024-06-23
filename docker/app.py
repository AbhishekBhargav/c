from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved model and scaler
model, scaler = joblib.load('logistic_regression_model_with_scaler.joblib')

# Initialize the Flask application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    # Convert data to numpy array
    try:
        new_data = np.array(data['features']).reshape(1, -1)
    except KeyError:
        return jsonify({'error': 'Invalid input data format'}), 400

    # Standardize the new data using the same scaler
    new_data_scaled = scaler.transform(new_data)

    # Predict the class of the new data
    prediction = model.predict(new_data_scaled)
    predicted_class = int(prediction[0])

    # Return the result as JSON
    return jsonify({'predicted_class': predicted_class})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5129, debug=True)
