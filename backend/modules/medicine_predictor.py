"""
medicine_predictor.py
Inference layer for Feature 3 (Predictive Analytics).

Loads the RandomForest models trained by train_predictor.py and, given a patient's
profile, scores every candidate drug/dose combination to recommend the best one --
replacing the old hardcoded JS scoring with real learned predictions.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

CANDIDATE_REGIMENS = [
    ("Metformin", 500), ("Metformin", 850), ("Metformin", 1000),
    ("Metformin", 1500), ("Metformin", 2000),
    ("Sitagliptin", 25), ("Sitagliptin", 50), ("Sitagliptin", 100),
    ("Empagliflozin", 10), ("Empagliflozin", 25),
]


class MedicinePredictor:
    _clf = None
    _reg = None
    _encoders = None
    _features = None

    @classmethod
    def load(cls):
        if cls._clf is None:
            cls._clf = joblib.load(os.path.join(MODEL_DIR, "responder_classifier.pkl"))
            cls._reg = joblib.load(os.path.join(MODEL_DIR, "reduction_regressor.pkl"))
            cls._encoders = joblib.load(os.path.join(MODEL_DIR, "predictor_encoders.pkl"))
            cls._features = joblib.load(os.path.join(MODEL_DIR, "predictor_features.pkl"))
        return cls._clf, cls._reg, cls._encoders, cls._features

    @classmethod
    def _encode_row(cls, row: dict, encoders: dict) -> dict:
        row = dict(row)
        for col, le in encoders.items():
            val = str(row.get(col, le.classes_[0]))
            if val not in le.classes_:
                val = le.classes_[0]
            row[col] = int(le.transform([val])[0])
        return row

    @classmethod
    def predict_best_medicine(cls, patient: dict, top_n: int = 3):
        """
        patient: dict with keys age, bmi, hba1c_baseline, fasting_glucose_mg_dl,
                 gender, diet_adherence, medication_adherence, lifestyle_activity
        Returns a ranked list of candidate regimens with predicted outcomes.
        """
        clf, reg, encoders, features = cls.load()

        results = []
        for drug, dose in CANDIDATE_REGIMENS:
            row = dict(patient)
            row["drug_name"] = drug
            row["dose_mg"] = dose
            row.setdefault("gender", "Female")
            row.setdefault("diet_adherence", "Moderate")
            row.setdefault("medication_adherence", "Moderate")
            row.setdefault("lifestyle_activity", "Moderate")

            enc_row = cls._encode_row(row, encoders)
            X = pd.DataFrame([enc_row])[features]

            predicted_reduction = float(reg.predict(X)[0])
            responder_pred = clf.predict(X)[0]
            responder_proba = dict(zip(clf.classes_, clf.predict_proba(X)[0].round(3)))

            # Composite score: weight predicted reduction + confidence of being a responder
            responder_conf = float(responder_proba.get("Responder", 0)) + \
                              0.5 * float(responder_proba.get("Partial Responder", responder_proba.get("Partial", 0)))
            score = round(min(100, max(0, predicted_reduction * 35 + responder_conf * 40 + 20)), 1)

            results.append({
                "drug": drug,
                "dose_mg": dose,
                "predicted_hba1c_reduction": round(predicted_reduction, 2),
                "predicted_responder_status": responder_pred,
                "responder_probabilities": responder_proba,
                "score": score,
            })

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_n]

    @classmethod
    def feature_importance(cls):
        clf, _, _, features = cls.load()
        importances = dict(zip(features, clf.feature_importances_.round(3)))
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
