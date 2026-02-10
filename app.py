from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------
# Load trained model & encoder
# ----------------------------
with open("model/churn_model.pkl", "rb") as f:
    model, le = pickle.load(f)

# ----------------------------
# API Status route
# ----------------------------
@app.route("/")
def api_home():
    return "Churn Prediction API Running"

# ----------------------------
# HTML UI route
# ----------------------------
@app.route("/ui")
def html_home():
    return render_template("index.html")

# ----------------------------
# Prediction route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Convert gender using LabelEncoder
    gender = le.transform([data["gender"]])[0]

    # Create feature array
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

    # Make prediction
    prediction = model.predict(features)[0]

    return jsonify({"churn_prediction": int(prediction)})


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
