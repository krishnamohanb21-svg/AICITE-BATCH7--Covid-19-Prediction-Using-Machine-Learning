from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
model_path = "covid_model.pkl"

if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    model = None
    print("Model file not found!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return render_template("result.html", prediction_text="Model file not found!")

        features = [
            int(request.form.get("breathing_problem", 0)),
            int(request.form.get("fever", 0)),
            int(request.form.get("dry_cough", 0)),
            int(request.form.get("sore_throat", 0)),
            int(request.form.get("running_nose", 0)),
            int(request.form.get("asthma", 0)),
            int(request.form.get("chronic_lung", 0)),
            int(request.form.get("headache", 0)),
            int(request.form.get("heart_disease", 0)),
            int(request.form.get("diabetes", 0)),
            int(request.form.get("hypertension", 0)),
            int(request.form.get("fatigue", 0)),
            int(request.form.get("gastrointestinal", 0)),
            int(request.form.get("abroad_travel", 0)),
            int(request.form.get("contact_covid", 0)),
            int(request.form.get("large_gathering", 0)),
            int(request.form.get("public_places", 0)),
            int(request.form.get("family_public", 0)),
            int(request.form.get("wearing_masks", 0)),
            int(request.form.get("sanitization", 0))
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)

        if prediction[0] == 1:
            result = "⚠ COVID-19 Positive"
        else:
            result = "✅ COVID-19 Negative"

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return render_template("result.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

