"""
generate_synthetic_patients.py

Generates synthetic patients using the trained CTGAN model.

KEY FIX: Removed MinMaxScaler inverse_transform.
         CTGAN outputs data in the original scale — no rescaling needed.
         dose_mg is handled as categorical string → converted back to int.
         Medical constraints applied AFTER generation.
"""

import pandas as pd
import numpy as np
import joblib
from ctgan import CTGAN

# ─────────────────────────────────────────────────────────────
# Valid dose values per drug (from drug labelling)
# ─────────────────────────────────────────────────────────────
VALID_DOSES = {
    "Metformin":     [500, 850, 1000, 1500, 2000],
    "Sitagliptin":   [25, 50, 100],
    "Empagliflozin": [10, 25],
    "None":          [0],
}

# Medical bounds for post-generation clipping
MEDICAL_BOUNDS = {
    "age":                        (30,  79),
    "bmi":                        (22.0, 45.0),
    "hba1c_baseline":             (6.5, 12.0),
    "hba1c_3_months":             (4.5, 14.0),
    "hba1c_6_months":             (4.5, 14.0),
    "hba1c_12_months":            (4.5, 14.0),
    "hba1c_24_months":            (4.5, 14.0),
    "hba1c_36_months":            (4.5, 14.0),
    "fasting_glucose_mg_dl":      (70,  350),
    "postprandial_glucose_mg_dl": (100, 450),
    "cholesterol_mg_dl":          (100, 350),
    "triglycerides_mg_dl":        (50,  500),
    "systolic_bp":                (90,  180),
    "diastolic_bp":               (55,  105),
    "diabetes_risk_probability_3m":  (0.05, 0.95),
    "diabetes_risk_probability_6m":  (0.05, 0.95),
    "diabetes_risk_probability_12m": (0.05, 0.95),
    "diabetes_risk_probability_24m": (0.05, 0.95),
    "diabetes_risk_probability_36m": (0.05, 0.95),
}


def load_model():
    model          = CTGAN.load("models/ctgan_model.pkl")
    columns        = joblib.load("models/columns.pkl")
    categorical_cols = joblib.load("models/categorical_cols.pkl")
    numeric_cols   = joblib.load("models/numeric_cols.pkl")
    data_ranges    = joblib.load("models/data_ranges.pkl")
    return model, columns, categorical_cols, numeric_cols, data_ranges


def fix_generated_samples(df):
    """
    Post-processing fixes after CTGAN generation:
    1. Convert dose_mg from string back to int
    2. Fix any invalid drug-dose combinations
    3. Clip numeric columns to medical bounds
    4. Fix discrete columns (age to int, etc.)
    5. Reassign patient IDs
    """
    df = df.copy()

    # ── dose_mg: string → int ─────────────────────────────────
    if "dose_mg" in df.columns:
        df["dose_mg"] = pd.to_numeric(df["dose_mg"], errors="coerce").fillna(0).round().astype(int)

    # ── validate drug-dose combinations ──────────────────────
    if "drug_name" in df.columns and "dose_mg" in df.columns:
        for idx, row in df.iterrows():
            drug = str(row.get("drug_name", "None"))
            dose = int(row.get("dose_mg", 0))
            if drug in VALID_DOSES:
                valid = VALID_DOSES[drug]
                if dose not in valid:
                    # Snap to nearest valid dose
                    df.at[idx, "dose_mg"] = min(valid, key=lambda x: abs(x - dose))
            else:
                df.at[idx, "dose_mg"] = 0

    # ── clip numeric columns to medical bounds ────────────────
    for col, (lo, hi) in MEDICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(lo, hi)

    # ── integer columns ───────────────────────────────────────
    int_cols = ["age", "fasting_glucose_mg_dl", "postprandial_glucose_mg_dl",
                "cholesterol_mg_dl", "triglycerides_mg_dl", "systolic_bp", "diastolic_bp"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].round().astype(int)

    # ── 1-decimal float columns ───────────────────────────────
    float1_cols = ["bmi", "hba1c_baseline", "hba1c_3_months", "hba1c_6_months",
                   "hba1c_12_months", "hba1c_24_months", "hba1c_36_months"]
    for col in float1_cols:
        if col in df.columns:
            df[col] = df[col].round(1)

    # ── 3-decimal risk probabilities ─────────────────────────
    risk_cols = [c for c in df.columns if "diabetes_risk_probability" in c]
    for col in risk_cols:
        df[col] = df[col].round(3)

    # ── fill any NaN categoricals with mode ───────────────────
    cat_defaults = {
        "drug_name": "Metformin",
        "treatment_group": "Treatment",
        "gender": "Female",
        "medication_adherence": "Moderate",
        "diet_adherence": "Moderate",
        "lifestyle_activity": "Moderate",
        "responder_status": "Minimal Responder",
        "adverse_event": "None",
    }
    for col, default in cat_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    # ── reassign unique patient IDs ───────────────────────────
    df["patient_id"] = [f"S{str(i + 1).zfill(5)}" for i in range(len(df))]

    return df


def recalculate_responder_status(df):
    """Recalculate responder status from generated HbA1c values."""
    if "hba1c_baseline" in df.columns and "hba1c_12_months" in df.columns:
        reduction = df["hba1c_baseline"] - df["hba1c_12_months"]
        conditions = [
            reduction >= 1.5,
            reduction >= 1.0,
            reduction >= 0.5,
            reduction >= 0.0,
        ]
        choices = [
            "Excellent Responder",
            "Good Responder",
            "Partial Responder",
            "Minimal Responder",
        ]
        df["responder_status"] = np.select(conditions, choices, default="Non-Responder")
    return df


def recalculate_risk_probabilities(df):
    """Recalculate diabetes risk probabilities from generated HbA1c values."""
    for months, col in [(3, "hba1c_3_months"), (6, "hba1c_6_months"),
                        (12, "hba1c_12_months"), (24, "hba1c_24_months"),
                        (36, "hba1c_36_months")]:
        risk_col = f"diabetes_risk_probability_{months}m"
        if col in df.columns and risk_col in df.columns:
            df[risk_col] = df[col].apply(
                lambda h: round(float(np.clip(1 / (1 + np.exp(-(h - 7.5) / 1.2)), 0.05, 0.95)), 3)
            )
    return df


# ─────────────────────────────────────────────────────────────
# READYMADE GENERATION
# ─────────────────────────────────────────────────────────────

def generate_readymade(n=10000):
    """Generate n synthetic patients using trained CTGAN."""
    model, columns, categorical_cols, numeric_cols, data_ranges = load_model()

    print(f"Generating {n} patients with CTGAN...")
    samples = model.sample(n)
    print(f"  Raw samples shape: {samples.shape}")

    samples = fix_generated_samples(samples)
    samples = recalculate_responder_status(samples)
    samples = recalculate_risk_probabilities(samples)

    # Ensure all original columns are present, in order
    for col in columns:
        if col not in samples.columns:
            samples[col] = np.nan
    samples = samples[[c for c in columns if c in samples.columns]]

    print(f"  Final samples shape: {samples.shape}")
    return samples


# ─────────────────────────────────────────────────────────────
# CUSTOM GENERATION with constraints
# ─────────────────────────────────────────────────────────────

def generate_custom(n=10000, constraints=None):
    """Generate patients then filter by user constraints."""
    model, columns, categorical_cols, numeric_cols, data_ranges = load_model()

    # Generate 3× to account for filtering losses
    oversample = n * 3
    print(f"Generating {oversample} patients (3× for constraint filtering)...")
    samples = model.sample(oversample)
    samples = fix_generated_samples(samples)

    if constraints:
        original_count = len(samples)
        for col, (min_val, max_val) in constraints.items():
            if col in samples.columns:
                samples[col] = pd.to_numeric(samples[col], errors="coerce")
                samples = samples[samples[col].between(min_val, max_val)]
        print(f"  Filtered: {original_count} → {len(samples)} patients")

        # If still not enough, generate more
        while len(samples) < n:
            extra = model.sample(n * 2)
            extra = fix_generated_samples(extra)
            for col, (min_val, max_val) in constraints.items():
                if col in extra.columns:
                    extra[col] = pd.to_numeric(extra[col], errors="coerce")
                    extra = extra[extra[col].between(min_val, max_val)]
            samples = pd.concat([samples, extra], ignore_index=True)
            print(f"  After top-up: {len(samples)} patients")

    samples = samples.head(n).reset_index(drop=True)
    samples = recalculate_responder_status(samples)
    samples = recalculate_risk_probabilities(samples)

    # Reassign IDs after filtering
    samples["patient_id"] = [f"S{str(i + 1).zfill(5)}" for i in range(len(samples))]

    for col in columns:
        if col not in samples.columns:
            samples[col] = np.nan
    samples = samples[[c for c in columns if c in samples.columns]]

    return samples


if __name__ == "__main__":
    print("=" * 50)
    print("Synthetic Patient Generator (CTGAN)")
    print("=" * 50)

    df = generate_readymade(20)
    print(f"\nGenerated {len(df)} patients")

    display = ["patient_id", "age", "bmi", "hba1c_baseline", "hba1c_12_months",
               "drug_name", "dose_mg", "treatment_group",
               "medication_adherence", "responder_status"]
    avail = [c for c in display if c in df.columns]
    print(df[avail].head(10).to_string())

    print("\nHbA1c trajectory:")
    for col in ["hba1c_baseline", "hba1c_6_months", "hba1c_12_months", "hba1c_36_months"]:
        if col in df.columns:
            trt  = df[df["treatment_group"] == "Treatment"][col].mean()
            ctrl = df[df["treatment_group"] == "Control"][col].mean()
            print(f"  {col}: Treatment={trt:.2f}  Control={ctrl:.2f}")