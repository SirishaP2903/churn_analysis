from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model/churn_model.pkl", "rb") as f:
    model, le = pickle.load(f)


@app.route("/")
def home():
    return "Churn Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert gender
    gender = le.transform([data["gender"]])[0]

    features = np.array([[
        data["customer_id"],
        gender,
        data["age"],
        data["tenure"],
        data["balance"],
        data["products_number"],
        data["has_credit_card"],
        data["is_active_member"],
        data["estimated_salary"]
    ]])

    prediction = model.predict(features)[0]

    return jsonify({"churn_prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
