

from flask import Flask, request, render_template_string
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")


HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>K-Means Clustering AI</title>

<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
    background: #080d1a;
    color: white;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 25px;
}

.container {
    width: 100%;
    max-width: 540px;
    background: #111a2e;
    padding: 40px;
    border-radius: 22px;
    border: 1px solid #263653;
    box-shadow: 0 20px 60px rgba(0,0,0,0.4);
}

.top {
    text-align: center;
    margin-bottom: 35px;
}

.logo {
    font-size: 42px;
    margin-bottom: 12px;
}

h1 {
    font-size: 29px;
    margin-bottom: 10px;
    background: linear-gradient(90deg,#38bdf8,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    color: #94a3b8;
    font-size: 14px;
}

.input-group {
    margin-bottom: 22px;
}

label {
    display: block;
    margin-bottom: 10px;
    color: #cbd5e1;
    font-size: 14px;
    font-weight: 600;
}

input {
    width: 100%;
    padding: 16px;
    border-radius: 11px;
    border: 1px solid #334155;
    background: #0b1324;
    color: white;
    font-size: 16px;
    outline: none;
    transition: 0.3s;
}

input:focus {
    border-color: #38bdf8;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.12);
}

button {
    width: 100%;
    padding: 17px;
    border: none;
    border-radius: 11px;
    background: linear-gradient(90deg,#0284c7,#7c3aed);
    color: white;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s;
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(56,189,248,0.25);
}

.result {
    margin-top: 28px;
    padding: 25px;
    background: #0b2230;
    border: 1px solid #155e75;
    border-radius: 15px;
    text-align: center;
    animation: appear 0.5s ease;
}

.result h2 {
    color: #67e8f9;
    font-size: 20px;
    margin-bottom: 12px;
}

.cluster {
    font-size: 43px;
    font-weight: bold;
    color: #38bdf8;
    margin: 12px 0;
}

.error {
    margin-top: 20px;
    padding: 15px;
    background: #451a1a;
    border-radius: 10px;
    color: #fca5a5;
}

.info {
    display: flex;
    justify-content: space-between;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #263653;
    color: #94a3b8;
    font-size: 12px;
}

.footer {
    text-align: center;
    margin-top: 25px;
    color: #64748b;
    font-size: 12px;
}

@keyframes appear {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
</head>

<body>

<div class="container">

    <div class="top">
        <div class="logo">🧠</div>
        <h1>K-Means Clustering</h1>
        <p class="subtitle">
            Intelligent Machine Learning Prediction System
        </p>
    </div>

    <form method="POST">

        <div class="input-group">
            <label>Feature 1</label>
            <input
                type="number"
                step="any"
                name="feature1"
                placeholder="Enter Feature 1 value"
                value="{{ feature1 if feature1 is not none else '' }}"
                required>
        </div>

        <div class="input-group">
            <label>Feature 2</label>
            <input
                type="number"
                step="any"
                name="feature2"
                placeholder="Enter Feature 2 value"
                value="{{ feature2 if feature2 is not none else '' }}"
                required>
        </div>

        <button type="submit">
            ✨ Predict Cluster
        </button>

    </form>

    {% if prediction is not none %}
    <div class="result">
        <h2>Prediction Successful!</h2>
        <div class="cluster">Cluster {{ prediction }}</div>
        <p>Your input belongs to Cluster {{ prediction }}</p>
    </div>
    {% endif %}

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

    <div class="info">
        <span>Algorithm: K-Means</span>
        <span>Total Clusters: 3</span>
    </div>

    <div class="footer">
        Built with Python, Flask & Scikit-Learn
    </div>

</div>

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None
    feature1 = None
    feature2 = None

    if request.method == "POST":

        try:
            feature1 = float(request.form["feature1"])
            feature2 = float(request.form["feature2"])

            if not np.isfinite([feature1, feature2]).all():
                raise ValueError("Please enter valid finite numbers.")

            input_data = pd.DataFrame(
                [[feature1, feature2]],
                columns=["Feature 1", "Feature 2"]
            )

            # Apply the same scaling used during training
            scaled_data = scaler.transform(input_data)

            # Predict cluster
            prediction = int(kmeans.predict(scaled_data)[0])

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "Prediction failed. Please check your input and model files."

    return render_template_string(
        HTML,
        prediction=prediction,
        error=error,
        feature1=feature1,
        feature2=feature2
    )


if __name__ == "__main__":
    app.run(debug=True)
