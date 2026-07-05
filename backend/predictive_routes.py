"""
predictive_routes.py
Flask blueprint for Feature 3, backed by the trained RandomForest models
instead of hardcoded JS scoring rules.

Register in app.py with:
    from routes.predictive_routes import predictive_bp
    app.register_blueprint(predictive_bp)
"""

from flask import Blueprint, request, jsonify
import traceback
from modules.medicine_predictor import MedicinePredictor

predictive_bp = Blueprint("predictive", __name__)


@predictive_bp.route("/api/predict-medicine", methods=["POST"])
def predict_medicine():
    try:
        data = request.json or {}

        patient = {
            "age": float(data.get("age", 55)),
            "bmi": float(data.get("bmi", 30)),
            "hba1c_baseline": float(data.get("hba1c", 8.0)),
            "fasting_glucose_mg_dl": float(data.get("glucose", 150)),
            "gender": data.get("gender", "Female"),
            "diet_adherence": data.get("diet_adherence", "Moderate"),
            "medication_adherence": data.get("medication_adherence", "Moderate"),
            "lifestyle_activity": data.get("lifestyle_activity", "Moderate"),
        }

        results = MedicinePredictor.predict_best_medicine(patient, top_n=5)
        importances = MedicinePredictor.feature_importance()

        return jsonify({
            "success": True,
            "best_regimen": results[0],
            "ranked_candidates": results,
            "model_feature_importance": importances,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
