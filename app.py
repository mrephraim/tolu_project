from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained models and scaler
cancer_model = joblib.load("cancer_model.joblib")
biomarker_model = joblib.load("biomarker_model.joblib")
scaler = joblib.load("scaler.joblib")

# Biomarker mapping
BIOMARKER_MAP = {
    1: "bTMB",
    2: "EGFR",
    3: "ALK",
    4: "CFDNA",
    5: "ESO"
}

# Expected feature order
FEATURES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE",
    "CHRONIC DISEASE", "FATIGUE", "ALLERGY", "WHEEZING", "ALCOHOL CONSUMING",
    "COUGHING", "SHORTNESS OF BREATH", "SWALLOWING DIFFICULTY", "CHEST PAIN"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/model-service")
def model():
    return render_template("services.html")

@app.route("/about-project")
def project():
    return render_template("about.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input data
        input_data = [float(request.form.get(feature, 0)) for feature in FEATURES]

        # Convert to DataFrame
        user_data = pd.DataFrame([input_data], columns=FEATURES)

        # Standardize input data
        user_data_scaled = scaler.transform(user_data)

        # Make predictions
        cancer_prediction = cancer_model.predict(user_data_scaled)[0]

        if cancer_prediction == 1:
            biomarker_prediction = biomarker_model.predict(user_data_scaled)[0]
            biomarker = BIOMARKER_MAP.get(int(biomarker_prediction), "Unknown")
        else:
            biomarker = "N/A"

        result = {
            "lung_cancer_status": "Lung cancer is positive or may be positive." if cancer_prediction == 1 else "Lung cancer is Negative",
            "biomarker": biomarker
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
