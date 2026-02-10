from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        features = [
            int(data["gender"]),
            float(data["age"]),
            float(data["tenure"]),
            float(data["balance"]),
            float(data["products_number"]),
            int(data["has_credit_card"]),
            int(data["is_active_member"]),
            float(data["estimated_salary"]),
        ]

        prediction = model.predict([features])[0]

        result = "Customer will churn ❌" if prediction == 1 else "Customer will stay ✅"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return jsonify({"error": str(e)})


# Run app
if __name__ == "__main__":
    app.run(debug=True)