from flask import Flask, request, jsonify
import joblib
import numpy as np

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
    X_new = np.array(data['features'])
    y_new = np.array(data['labels'])

    # Fine-tune the model
    model.fit(X_new, y_new)

    # Save the updated model
    joblib.dump(model, 'random_forest_model.pkl')
    return jsonify({'message': 'Model fine-tuned successfully'})

if __name__ == '__main__':
    app.run(debug=True)
