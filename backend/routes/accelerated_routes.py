"""
accelerated_routes.py
Flask blueprint for Feature 4, backed by a Markov chain disease-progression
model instead of plain client-side percentage counting.

Register in app.py with:
    from routes.accelerated_routes import accelerated_bp
    app.register_blueprint(accelerated_bp)
"""

from flask import Blueprint, request, jsonify
import pandas as pd
import traceback
from modules.markov_progression import project_timeline

accelerated_bp = Blueprint("accelerated", __name__)


@accelerated_bp.route("/api/markov-projection", methods=["POST"])
def markov_projection():
    try:
        data = request.json or {}
        patients = data.get("patients", [])
        extra_steps = int(data.get("extra_steps", 2))  # how far beyond 36m to project

        if not patients:
            return jsonify({"success": False, "error": "No patients received"}), 400

        df = pd.DataFrame(patients)
        required_cols = ["hba1c_baseline", "hba1c_3_months", "hba1c_6_months",
                          "hba1c_12_months", "hba1c_24_months", "hba1c_36_months",
                          "treatment_group"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return jsonify({"success": False, "error": f"Missing columns: {missing}"}), 400

        for col in required_cols[:-1]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        projection = project_timeline(df, extra_steps=extra_steps)

        return jsonify({
            "success": True,
            "states": ["Controlled", "Borderline", "Uncontrolled"],
            "projection": projection,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
