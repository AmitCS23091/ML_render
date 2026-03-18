from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return jsonify({
        "message": "Iris ML API is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]

        if len(features) != 4:
            return jsonify({"error": "Provide exactly 4 values"}), 400

        features = np.array(features).reshape(1, -1)

        # Apply scaling
        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return jsonify({
        "message": "Iris ML API is running 🚀"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400

        features = data["features"]

        if len(features) != 4:
            return jsonify({"error": "Provide exactly 4 values"}), 400

        features = np.array(features).reshape(1, -1)

        # Apply scaling
        features = scaler.transform(features)

        prediction = model.predict(features)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)