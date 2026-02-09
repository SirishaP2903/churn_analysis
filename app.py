from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# load the model

with open("model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    scaled_data = scaler.transform([data])
    prediction = model.predict(scaled_data)[0]
    result = "Customer is likely to churn" if prediction == 1 else "Cusotmer will stay"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
