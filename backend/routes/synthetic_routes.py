"""
synthetic_routes.py
FIXED: Uses corrected generator with clean separation and random mixing
"""

from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import random
from modules.generate_data import generate_patients  # Import from FIXED file

synthetic_bp = Blueprint("synthetic", __name__)


@synthetic_bp.route("/generate", methods=["POST"])
def generate():
    data = request.json
    population_params = data.get("population_params", {})
    n = min(population_params.get("num_patients", 1000), 10000)
    dataset_type = data.get("dataset_type", "readymade")
    constraints = None

    if dataset_type == "custom":
        constraints = {}
        if "age_range" in population_params:
            constraints["age_range"] = {
                "min": population_params["age_range"]["min"],
                "max": population_params["age_range"]["max"]
            }
        if "bmi_range" in population_params:
            constraints["bmi_range"] = {
                "min": population_params["bmi_range"]["min"],
                "max": population_params["bmi_range"]["max"]
            }
        if "hba1c_range" in population_params:
            constraints["hba1c_range"] = {
                "min": population_params["hba1c_range"]["min"],
                "max": population_params["hba1c_range"]["max"]
            }
        if "glucose_range" in population_params:
            constraints["glucose_range"] = {
                "min": population_params["glucose_range"]["min"],
                "max": population_params["glucose_range"]["max"]
            }

    try:
        print(f"Generating {n} patients with type: {dataset_type}")
        
        # Generate using FIXED generator
        df = generate_patients(n=n, constraints=constraints, dataset_type=dataset_type)
        
        # Ensure patient IDs are sequential
        df["patient_id"] = [f"P{str(i+1).zfill(5)}" for i in range(len(df))]
        
        # Verify clean separation
        treatment_none = len(df[(df["treatment_group"] == "Treatment") & (df["drug_name"] == "None")])
        control_drug = len(df[(df["treatment_group"] == "Control") & (df["drug_name"] != "None")])
        
        print(f"✅ VERIFICATION - Treatment with None: {treatment_none} (should be 0)")
        print(f"✅ VERIFICATION - Control with drugs: {control_drug} (should be 0)")
        print(f"✅ Treatment: {len(df[df['treatment_group'] == 'Treatment'])}")
        print(f"✅ Control: {len(df[df['treatment_group'] == 'Control'])}")

        # Convert numpy types for JSON
        records = []
        for record in df.to_dict("records"):
            clean = {}
            for k, v in record.items():
                if isinstance(v, (np.int64, np.int32, np.int16)):
                    clean[k] = int(v)
                elif isinstance(v, (np.float64, np.float32)):
                    clean[k] = None if pd.isna(v) else float(v)
                else:
                    clean[k] = v
            records.append(clean)

        return jsonify({
            "success": True,
            "data": records,
            "total_patients": len(records),
            "validation": {
                "treatment_count": len(df[df["treatment_group"] == "Treatment"]),
                "control_count": len(df[df["treatment_group"] == "Control"]),
                "treatment_with_none": treatment_none,
                "control_with_drugs": control_drug
            }
        })

    except Exception as e:
        print(f"Generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500