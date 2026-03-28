from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        income = float(request.form['income'])
        credit_score = float(request.form['credit_score'])
        loan_amount = float(request.form['loan_amount'])
        employment = int(request.form['employment'])

        # Simple logic (no ML for now)
        if income > 30000 and credit_score > 700 and employment == 1:
            result = "Loan Approved ✅"
            status = "success"
        else:
            result = "Loan Rejected ❌"
            status = "failure"

        return render_template('result.html', prediction=result, status=status)

    return render_template('index.html')

@app.route('/healthz')
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)