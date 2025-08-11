# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import joblib

app = Flask(__name__)
CORS(app)

# --- load team data from team_data.json (simple, no firebase-admin needed)
DATA_PATH = os.path.join(os.path.dirname(__file__), "team_data.json")
try:
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        TEAM_DATA = json.load(fh)
except Exception as e:
    print("Failed to load team_data.json:", e)
    TEAM_DATA = []

# optional ML model (place model.pkl in repo root if you want)
MODEL = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
if os.path.exists(MODEL_PATH):
    try:
        MODEL = joblib.load(MODEL_PATH)
        print("Loaded model.pkl")
    except Exception as e:
        print("Failed to load model.pkl:", e)
        MODEL = None

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route("/suggest-team", methods=["POST"])
def suggest_team():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    # simple matching mode (preferred): client sends "skills": [..]
    skills = data.get("skills") or []
    top_n = int(data.get("top_n", 3))

    if isinstance(skills, list) and len(skills) > 0:
        # normalize
        skills_lower = set([s.strip().lower() for s in skills if isinstance(s, str)])

        results = []
        for member in TEAM_DATA:
            m_skills = member.get("skills", [])
            m_skills_lower = set([s.strip().lower() for s in m_skills if isinstance(s, str)])
            matched = list(skills_lower.intersection(m_skills_lower))
            score = len(matched)
            results.append({
                "name": member.get("name"),
                "role": member.get("role"),
                "matched_skills": matched,
                "score": score,
                "extra": member.get("extra", {})
            })

        # sort by score desc then by name
        results_sorted = sorted(results, key=lambda x: (-x["score"], x["name"] or ""))
        # pick top_n; if none matched (score==0), still return top_n fallback
        suggested = [r for r in results_sorted if r["score"] > 0][:top_n]
        if not suggested:
            suggested = results_sorted[:top_n]

        return jsonify({"suggested_team": suggested}), 200

    # optional: ML path
    if data.get("use_ml") and MODEL is not None:
        features = data.get("features")
        if features is None:
            return jsonify({"error": "Missing 'features' for ML mode"}), 400
        try:
            pred = MODEL.predict([features])
            return jsonify({"prediction": pred.tolist()}), 200
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

    return jsonify({"error": "Provide 'skills' list (POST JSON)"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
