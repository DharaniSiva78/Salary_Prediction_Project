from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl'))
model = joblib.load(MODEL_PATH)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    # Assume preprocessing is built into model pipeline
    prediction = model.predict(df)
    return jsonify({'predicted_ctc': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
