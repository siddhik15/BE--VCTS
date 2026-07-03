"""
medical_validator.py
Medical validation against ADA 2025 Standards of Care.
FIX: Updated column names (bmi not bmi_baseline, hba1c_baseline not baseline_hba1c).
FIX: Standard mode now uses correct T2D clinical trial thresholds.
"""

import pandas as pd
import numpy as np


class MedicalValidator:
    """Validate synthetic patient data against ADA 2025 clinical guidelines."""

    # ADA 2025 HbA1c targets
    ADA_TARGETS = {
        "hba1c": {
            "diagnosis_min":   6.5,   # % — T2D diagnosis threshold
            "target":          7.0,   # % — treatment target
            "poor_control":    8.0,   # % — poor glycaemic control
            "very_poor":      10.0,   # % — very poor, intensification needed
        },
        "fasting_glucose": {
            "normal_max":    100,     # mg/dL
            "prediabetes":   125,     # mg/dL
            "t2d_min":       126,     # mg/dL
        },
        "blood_pressure": {
            "systolic_target":  130,  # mmHg
            "diastolic_target":  80,  # mmHg
        },
    }

    # Clinical trial inclusion criteria (standard level)
    TRIAL_CRITERIA = {
        "age":   {"min": 30, "max": 80},
        "bmi":   {"min": 22.0, "max": 45.0},
        "hba1c": {"min": 6.5,  "max": 12.0},
    }

    # Drug-dose valid combinations
    VALID_DOSES = {
        "Metformin":     [500, 850, 1000, 1500, 2000],
        "Sitagliptin":   [25, 50, 100],
        "Empagliflozin": [10, 25],
        "None":          [0],
    }

    @classmethod
    def validate_patient(cls, patient_data):
        """
        Validate a single patient record.
        Returns: (is_valid: bool, issues: list[str])
        """
        issues = []
        criteria = cls.TRIAL_CRITERIA

        # Age
        age = patient_data.get("age")
        if age is not None:
            try:
                age = float(age)
                if age < criteria["age"]["min"]:
                    issues.append(f"Age {age:.0f} below minimum ({criteria['age']['min']})")
                if age > criteria["age"]["max"]:
                    issues.append(f"Age {age:.0f} exceeds maximum ({criteria['age']['max']})")
            except (TypeError, ValueError):
                pass

        # BMI (column is 'bmi' in new schema)
        bmi = patient_data.get("bmi") or patient_data.get("bmi_baseline") or patient_data.get("BMI")
        if bmi is not None:
            try:
                bmi = float(bmi)
                if bmi < criteria["bmi"]["min"]:
                    issues.append(f"BMI {bmi:.1f} below minimum ({criteria['bmi']['min']})")
                if bmi > criteria["bmi"]["max"]:
                    issues.append(f"BMI {bmi:.1f} exceeds maximum ({criteria['bmi']['max']})")
            except (TypeError, ValueError):
                pass

        # HbA1c (column is 'hba1c_baseline' in new schema)
        hba1c = (patient_data.get("hba1c_baseline") or
                 patient_data.get("baseline_hba1c") or
                 patient_data.get("HbA1c"))
        if hba1c is not None:
            try:
                hba1c = float(hba1c)
                if hba1c < cls.ADA_TARGETS["hba1c"]["diagnosis_min"]:
                    issues.append(
                        f"HbA1c {hba1c:.1f}% below T2D diagnosis threshold "
                        f"({cls.ADA_TARGETS['hba1c']['diagnosis_min']}%)"
                    )
                if hba1c > criteria["hba1c"]["max"]:
                    issues.append(f"HbA1c {hba1c:.1f}% exceeds maximum ({criteria['hba1c']['max']}%)")
            except (TypeError, ValueError):
                pass

        # Drug-dose validation
        drug = patient_data.get("drug_name", "None")
        dose = patient_data.get("dose_mg", 0)
        if drug and str(drug) in cls.VALID_DOSES:
            try:
                dose = int(float(dose))
                if dose not in cls.VALID_DOSES[str(drug)]:
                    issues.append(
                        f"Invalid dose {dose}mg for {drug}. "
                        f"Valid: {cls.VALID_DOSES[str(drug)]}"
                    )
            except (TypeError, ValueError):
                pass

        return len(issues) == 0, issues

    @classmethod
    def validate_trial_population(cls, df, trial_name="Clinical Trial"):
        """Validate entire cohort and return summary report."""
        report = {
            "trial_name":       trial_name,
            "total_patients":   len(df),
            "valid_patients":   0,
            "invalid_patients": 0,
            "issues":           [],
            "statistics":       {},
        }

        for idx, row in df.iterrows():
            is_valid, issues = cls.validate_patient(row.to_dict())
            if is_valid:
                report["valid_patients"] += 1
            else:
                report["invalid_patients"] += 1
                if len(report["issues"]) < 50:
                    pid = row.get("patient_id", idx)
                    report["issues"].extend([f"Patient {pid}: {iss}" for iss in issues])

        # Population statistics
        base_col = "hba1c_baseline" if "hba1c_baseline" in df.columns else "baseline_hba1c"
        bmi_col  = "bmi" if "bmi" in df.columns else "bmi_baseline"

        if base_col in df.columns:
            vals = pd.to_numeric(df[base_col], errors="coerce").dropna()
            report["statistics"].update({
                "hba1c_mean": round(float(vals.mean()), 2),
                "hba1c_std":  round(float(vals.std()),  2),
                "hba1c_min":  round(float(vals.min()),  2),
                "hba1c_max":  round(float(vals.max()),  2),
                "pct_at_target": round(
                    float((vals <= cls.ADA_TARGETS["hba1c"]["target"]).mean() * 100), 1
                ),
            })

        if bmi_col in df.columns:
            bmi_vals = pd.to_numeric(df[bmi_col], errors="coerce").dropna()
            report["statistics"].update({
                "bmi_mean":   round(float(bmi_vals.mean()), 1),
                "pct_obese":  round(float((bmi_vals >= 30).mean() * 100), 1),
            })

        return report

    @classmethod
    def validate_with_level(cls, df, trial_name="Clinical Trial", level="standard"):
        """
        Validate with different strictness levels.
        level: "strict" | "standard" | "relaxed"
        """
        original = {k: v.copy() for k, v in cls.TRIAL_CRITERIA.items()}

        if level == "strict":
            cls.TRIAL_CRITERIA["age"]["max"]   = 75
            cls.TRIAL_CRITERIA["bmi"]["min"]   = 27.0
            cls.TRIAL_CRITERIA["hba1c"]["min"] = 7.0
            cls.TRIAL_CRITERIA["hba1c"]["max"] = 10.5
        elif level == "relaxed":
            cls.TRIAL_CRITERIA["age"]["max"]   = 85
            cls.TRIAL_CRITERIA["bmi"]["min"]   = 20.0
            cls.TRIAL_CRITERIA["bmi"]["max"]   = 50.0
            cls.TRIAL_CRITERIA["hba1c"]["min"] = 6.0
            cls.TRIAL_CRITERIA["hba1c"]["max"] = 13.0
        else:  # standard
            cls.TRIAL_CRITERIA["age"]["max"]   = 80
            cls.TRIAL_CRITERIA["bmi"]["min"]   = 22.0
            cls.TRIAL_CRITERIA["hba1c"]["min"] = 6.5
            cls.TRIAL_CRITERIA["hba1c"]["max"] = 12.0

        report = cls.validate_trial_population(df, f"{trial_name} ({level})")
        report["validation_level"] = level
        report["criteria_used"] = {
            "age":   f"{cls.TRIAL_CRITERIA['age']['min']}–{cls.TRIAL_CRITERIA['age']['max']}",
            "bmi":   f"{cls.TRIAL_CRITERIA['bmi']['min']}–{cls.TRIAL_CRITERIA['bmi']['max']}",
            "hba1c": f"{cls.TRIAL_CRITERIA['hba1c']['min']}–{cls.TRIAL_CRITERIA['hba1c']['max']}%",
        }

        cls.TRIAL_CRITERIA = original
        return report