from flask import Flask, request, jsonify
import joblib
import numpy as np
from scipy.optimize import minimize

app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_model.pkl')

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/fine_tune', methods=['POST'])
def fine_tune():
    data = request.json
    features = np.array(data['features'])
    target = np.array(data['target'])

    # Fine-tune the model
    model.fit(features, target)

    # Save the updated model
    joblib.dump(model, 'random_forest_model.pkl')

    # Calculate mean and standard deviation for each feature
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0)
    initial_guess = feature_means + feature_stds

    # Function to get the probability of predicting 1
    def predict_proba(features):
        return model.predict_proba([features])[0, 1]

    try:
        # Optimize the features to maximize the probability of predicting 1
        result = minimize(lambda x: -predict_proba(x), initial_guess, method='Nelder-Mead')
        optimized_features = result.x
    except Exception as e:
        # In case of error, return the initial guess
        optimized_features = initial_guess

    return jsonify({
        'message': 'Model fine-tuned successfully',
        'optimized_features': optimized_features.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
