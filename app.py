from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
import json

# === Initialize Flask App ===
app = Flask(__name__)

# === Load Artifacts ===
ARTIFACTS_DIR = "artifacts"
model_reg_path = os.path.join(ARTIFACTS_DIR, "model_regression.joblib")
model_clf_path = os.path.join(ARTIFACTS_DIR, "model_classification.joblib")
encoders_path = os.path.join(ARTIFACTS_DIR, "label_encoders.json")

model_reg = joblib.load(model_reg_path)
model_clf = joblib.load(model_clf_path)

with open(encoders_path, "r") as f:
    label_encoders = json.load(f)

# Features
cat_features = ['generic_name', 'medical_condition', 'drug_classes',
                'rx_otc', 'pregnancy_category', 'csa']
num_features = [
    'activity','alcohol','no_of_reviews','side_effects_len','mc_desc_len',
    'has_hives','has_difficult_breathing','has_itching','has_rash','has_dizziness'
]

# === Routes ===
@app.route('/')
def index():
    # pass dropdown values directly (no integer mapping)
    options = {col: list(values.keys()) for col, values in label_encoders.items()}
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form = request.form
        input_data = pd.DataFrame([{
            "generic_name": form.get("generic_name"),
            "medical_condition": form.get("medical_condition"),
            "drug_classes": form.get("drug_classes"),
            "rx_otc": form.get("rx_otc"),
            "pregnancy_category": form.get("pregnancy_category"),
            "csa": form.get("csa"),
            "activity": float(form.get("activity", 0)) / 100.0,
            "alcohol": int(form.get("alcohol", 0)),
            "no_of_reviews": int(form.get("no_of_reviews", 0)),
            "side_effects_len": len(form.get("side_effects", "")),
            "mc_desc_len": len(form.get("medical_condition_description", "")),
            "has_hives": int("hives" in form.get("side_effects", "").lower()),
            "has_difficult_breathing": int("difficult breathing" in form.get("side_effects", "").lower()),
            "has_itching": int("itching" in form.get("side_effects", "").lower()),
            "has_rash": int("rash" in form.get("side_effects", "").lower()),
            "has_dizziness": int("dizziness" in form.get("side_effects", "").lower()),
        }])

        print("\nDEBUG CLEAN INPUT DATAFRAME:")
        print(input_data.dtypes)
        print(input_data)

        # let the pipeline handle encoding internally
        predicted_rating = model_reg.predict(input_data)[0]

        if predicted_rating >= 7:
            rating_class = "High"
        elif predicted_rating >= 4:
            rating_class = "Medium"
        else:
            rating_class = "Low"

        return render_template("result.html",
                               rating=round(float(predicted_rating), 2),
                               rating_class=rating_class)
    except Exception as e:
        return render_template("result.html", rating="Error", rating_class=str(e))

if __name__ == '__main__':
    app.run(debug=True)
