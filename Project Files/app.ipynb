from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
model = joblib.load('model/traffic_model.pkl')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        preds = model.predict(df)
        return render_template('result.html', predictions=preds.tolist())
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
