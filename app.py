# app.py
import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Firebase admin (optional - dynamic from Firestore)
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
CORS(app)

# ---------- Firestore init (optional) ----------
USE_FIRESTORE = False
db = None

if os.environ.get("FIREBASE_SERVICE_ACCOUNT"):
    try:
        cred_json = json.loads(os.environ["FIREBASE_SERVICE_ACCOUNT"])
        cred = credentials.Certificate(cred_json)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        USE_FIRESTORE = True
        app.logger.info("Firestore initialized from FIREBASE_SERVICE_ACCOUNT env var.")
    except Exception as e:
        app.logger.exception("Failed to init Firestore: %s", e)
        USE_FIRESTORE = False

# ---------- Load fallback local data ----------
TEAM_DATA = []
if os.path.exists("team_data.json"):
    try:
        with open("team_data.json", "r", encoding="utf-8") as f:
            TEAM_DATA = json.load(f)
    except Exception as e:
        app.logger.exception("Failed loading team_data.json: %s", e)
        TEAM_DATA = []

# ---------- Helpers ----------
def fetch_candidates_from_firestore():
    # reads students collection and returns list of dicts with name, role, skills
    candidates = []
    try:
        docs = db.collection("students").stream()
        for d in docs:
            data = d.to_dict()
            skills = data.get("skills", [])
            # normalize: allow both list or comma string
            if isinstance(skills, str):
                skills = [s.strip() for s in skills.split(",") if s.strip()]
            candidates.append({
                "name": data.get("name", ""),
                "role": data.get("role", ""),
                "skills": skills,
                "extra": {}
            })
    except Exception as e:
        app.logger.exception("Error fetching candidates from Firestore: %s", e)
    return candidates

def fetch_candidates():
    if USE_FIRESTORE and db is not None:
        c = fetch_candidates_from_firestore()
        if c:
            return c
    # fallback
    return TEAM_DATA

# ---------- Routes ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route("/suggest-team", methods=["POST"])
def suggest_team():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Invalid JSON payload"}), 400

    skills = payload.get("skills")
    top_n = int(payload.get("top_n", 3))

    if not skills or not isinstance(skills, (list, tuple)):
        return jsonify({"error": "Provide 'skills' list"}), 400

    candidates = fetch_candidates()
    if not candidates:
        return jsonify({"suggested_team": []}), 200

    # Prepare text data for TF-IDF
    cand_texts = []
    for c in candidates:
        s = c.get("skills", [])
        if isinstance(s, list):
            cand_texts.append(" ".join([str(x).lower() for x in s]))
        else:
            cand_texts.append(str(s).lower())

    user_text = " ".join([str(x).lower() for x in skills])

    try:
        vectorizer = TfidfVectorizer()
        all_docs = cand_texts + [user_text]
        mat = vectorizer.fit_transform(all_docs)  # shape (n_cand+1, n_feats)
        user_vec = mat[-1]
        cand_vecs = mat[:-1]
        sims = cosine_similarity(user_vec, cand_vecs).flatten()
    except Exception as e:
        app.logger.exception("Vectorization error: %s", e)
        # fallback: simple overlap count
        sims = []
        user_set = set([x.lower() for x in skills])
        for c in candidates:
            sset = set([str(x).lower() for x in c.get("skills", [])])
            inter = user_set.intersection(sset)
            sims.append(len(inter) / (len(sset) + 1e-6))

    results = []
    user_sk_lower = [x.lower() for x in skills]
    for idx, sim in enumerate(sims):
        c = candidates[idx]
        matched = [sk for sk in c.get("skills", []) if str(sk).lower() in user_sk_lower]
        results.append({
            "name": c.get("name", ""),
            "role": c.get("role", ""),
            "matched_skills": matched,
            "score": float(round(float(sim), 4))
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]
    return jsonify({"suggested_team": results}), 200

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
