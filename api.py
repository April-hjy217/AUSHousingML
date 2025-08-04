from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")

# Load the model at startup
with open(MODEL_PATH, "rb") as f:
    model = joblib.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
      "features": [val1, val2, ...]  # Must match model's feature order
    }
    """
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in JSON"}), 400
    features = data["features"]
    try:
        pred = model.predict([features])
        return jsonify({"prediction": float(pred[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Listen on all interfaces so Docker works
    app.run(host="0.0.0.0", port=8000, debug=True)
