from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/healthz')
def health():
    return "OK", 200

@app.route('/predict', methods=['POST'])
def predict():
    income = float(request.form['income'])
    credit_score = float(request.form['credit_score'])
    loan_amount = float(request.form['loan_amount'])
    employment = int(request.form['employment'])

    features = np.array([[income, credit_score, loan_amount, employment]])
    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]

    result = "Approved ✅" if prediction == 1 else "Not Approved ❌"

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)