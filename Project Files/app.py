from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        holiday = float(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hour'])

        # Combine into array for prediction
        input_features = np.array([[holiday, temp, rain, snow, year, month, day, hour]])

        # Scale the input
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_text = f"Estimated Traffic Volume: {int(prediction)} units"

        return render_template('index.html', prediction_text=prediction_text)

    except Exception as e:
        print("Prediction error:", e)
        return render_template('index.html', prediction_text="Oops! Something went wrong. Unable to estimate traffic volume with the given input. Try again.")

if __name__ == '__main__':
    app.run(debug=True)
