from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join("model", "traffic_model.pkl")  # Ensure the correct model path
model = joblib.load(MODEL_PATH)

def prepare_input(df):
    """Prepare incoming data to match training features."""
    return df  # No additional preparation needed for this dataset

# ----------------- ROUTES ----------------- #

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/application')
def application():
    return render_template('application.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        hour = int(request.form['hour'])
        temp = float(request.form['temp'])
        holiday = int(request.form['holiday'])
        weather_main = request.form['weather_main']

        input_df = pd.DataFrame({
            'hour': [hour],
            'temp': [temp],
            'holiday': [holiday],
            'weather_main': [weather_main]
        })

        prediction = model.predict(input_df)[0]

        return render_template(
            'result.html',
            prediction=round(prediction),
            hour=hour,
            temp=temp,
            holiday=holiday,
            weather_main=weather_main
        )
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict_file():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Ensure the input DataFrame has the correct columns
        required_columns = ['hour', 'temp', 'holiday', 'weather_main']
        if not all(col in df.columns for col in required_columns):
            return "Error: Input CSV must contain the following columns: " + ", ".join(required_columns)

        df_prepared = prepare_input(df)
        preds = model.predict(df_prepared)

        return render_template('result.html', predictions=preds.tolist())
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# ----------------- MAIN ----------------- #
if __name__ == '__main__':
    app.run(debug=True)