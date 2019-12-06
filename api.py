from flask import Flask, jsonify, request
from flask_cors import CORS
from predict_pipeline import PredictPipeline
import pandas as pd


app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
    return jsonify({"Message": "Hello Machine Learning World"})


@app.route('/model', methods=['POST'])
def model():
    input_data = pd.DataFrame.from_records([request.get_json()])
    pipeline = PredictPipeline('scaler.pkl', 'logreg_model.pkl')
    prediction, probability = pipeline.predict(input_data)
    return jsonify({'prediction': int(prediction), 'probability': float(probability)})


if __name__ == "__main__":
    app.run(debug=True)