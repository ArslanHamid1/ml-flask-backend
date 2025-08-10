from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Health check route
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

# Example: Team suggestion endpoint
@app.route("/suggest-team", methods=["POST"])
def suggest_team():
    data = request.get_json()
    skills = data.get("skills", [])

    if not skills:
        return jsonify({"error": "No skills provided"}), 400

    # Dummy AI logic (replace with real model later)
    team = ["Member A", "Member B", "Member C"]
    return jsonify({"suggested_team": team}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
