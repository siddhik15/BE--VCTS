"""
fix_csv.py
Run this ONCE from your project root to fix the CSV in place.
It will overwrite data/diabetes_dataset_fixed.csv with a clean version.

Usage:
    python fix_csv.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── find the CSV ──────────────────────────────────────────────
candidates = [
    "data/diabetes_dataset_fixed.csv",
    "data/diabetes_dataset.csv",
]
src = None
for c in candidates:
    if os.path.exists(c):
        src = c
        break

if not src:
    print("ERROR: Could not find CSV in data/. Place diabetes_dataset.csv in data/ first.")
    raise SystemExit(1)

print(f"Loading: {src}")
df = pd.read_csv(src)
print(f"  Shape: {df.shape}   NaNs before: {df.isnull().sum().sum()}")

# ── 1. Rename BMI → bmi ──────────────────────────────────────
if "BMI" in df.columns and "bmi" not in df.columns:
    df.rename(columns={"BMI": "bmi"}, inplace=True)
    print("  Renamed BMI → bmi")

# ── 2. Fill NaN categoricals (CTGAN crashes on NaN) ──────────
df["drug_name"]     = df["drug_name"].fillna("None")
df["adverse_event"] = df["adverse_event"].fillna("None")

# ── 3. Medical range fixes ────────────────────────────────────
mask = df["hba1c_baseline"] < 6.5
if mask.sum():
    df.loc[mask, "hba1c_baseline"] = np.round(np.random.uniform(6.5, 7.5, mask.sum()), 1)
    print(f"  Fixed {mask.sum()} HbA1c < 6.5")

mask = df["bmi"] < 22
if mask.sum():
    df.loc[mask, "bmi"] = np.round(np.random.uniform(22.0, 30.0, mask.sum()), 1)
    print(f"  Fixed {mask.sum()} BMI < 22")

mask = df["triglycerides_mg_dl"] < 50
if mask.sum():
    df.loc[mask, "triglycerides_mg_dl"] = np.random.randint(100, 200, mask.sum())
    print(f"  Fixed {mask.sum()} triglycerides < 50")

mask = df["diastolic_bp"] < 55
if mask.sum():
    df.loc[mask, "diastolic_bp"] = np.random.randint(60, 75, mask.sum())
    print(f"  Fixed {mask.sum()} diastolic BP < 55")

# ── 4. Recalculate HbA1c follow-ups with adherence model ─────
DRUG_DOSE_EFFICACY = {
    ("Metformin",500):(0.70,0.22),("Metformin",850):(1.00,0.25),
    ("Metformin",1000):(1.15,0.27),("Metformin",1500):(1.35,0.28),("Metformin",2000):(1.50,0.30),
    ("Sitagliptin",25):(0.42,0.15),("Sitagliptin",50):(0.62,0.18),("Sitagliptin",100):(0.82,0.22),
    ("Empagliflozin",10):(0.68,0.20),("Empagliflozin",25):(0.88,0.22),("None",0):(0.00,0.10),
}
MED  = {"Poor":0.35,"Moderate":0.65,"Good":1.00}
DIET = {"Poor":0.72,"Moderate":0.86,"Good":1.00}
LIFE = {"Sedentary":0.78,"Moderate":0.92,"Active":1.10}
CTRL_LIFE = {"Sedentary":-0.20,"Moderate":0.10,"Active":0.38}
CTRL_DIET = {"Poor":-0.12,"Moderate":0.05,"Good":0.18}

def hba1c_at_month(row, months):
    b    = row["hba1c_baseline"]
    drug = row["drug_name"]
    dose = int(row["dose_mg"])
    grp  = row["treatment_group"]
    med  = row["medication_adherence"]
    diet = row["diet_adherence"]
    life = row["lifestyle_activity"]

    if grp == "Control":
        tf     = min(1.0, months / 12.0)
        change = (CTRL_LIFE.get(life, 0) + CTRL_DIET.get(diet, 0)) * tf
        result = b - change + np.random.normal(0, 0.12)
    else:
        mr, sr = DRUG_DOSE_EFFICACY.get((drug, dose), (0.80, 0.25))
        sev    = 1.0 + max(0.0, (b - 8.0) * 0.05)
        if months <= 3:    t = months / 6.0
        elif months <= 6:  t = 0.5  + (months - 3)  / 12.0
        elif months <= 12: t = 1.0  + (months - 6)  / 30.0
        elif months <= 24: t = 1.20 + (months - 12) / 48.0
        else:              t = 1.45 + (months - 24) / 96.0
        red    = mr * MED.get(med,0.65) * DIET.get(diet,0.86) * LIFE.get(life,0.92) * sev * t
        result = b - red + np.random.normal(0, sr * 0.45)

    return round(float(np.clip(result, 4.5, 14.0)), 1)

print("  Recalculating HbA1c follow-up values...")
for months, col in [(3,"hba1c_3_months"),(6,"hba1c_6_months"),(12,"hba1c_12_months"),
                    (24,"hba1c_24_months"),(36,"hba1c_36_months")]:
    df[col] = df.apply(lambda r: hba1c_at_month(r, months), axis=1)

# ── 5. Recalculate responder status ──────────────────────────
red = df["hba1c_baseline"] - df["hba1c_12_months"]
df["responder_status"] = np.select(
    [red >= 1.5, red >= 1.0, red >= 0.5, red >= 0.0],
    ["Excellent Responder","Good Responder","Partial Responder","Minimal Responder"],
    default="Non-Responder"
)

# ── 6. Recalculate risk probabilities ────────────────────────
for months, hcol in [(3,"hba1c_3_months"),(6,"hba1c_6_months"),(12,"hba1c_12_months"),
                     (24,"hba1c_24_months"),(36,"hba1c_36_months")]:
    rcol = f"diabetes_risk_probability_{months}m"
    df[rcol] = df[hcol].apply(
        lambda h: round(float(np.clip(1/(1+np.exp(-(h-7.5)/1.2)), 0.05, 0.95)), 3)
    )

# ── 7. Recalculate fasting glucose ───────────────────────────
df["fasting_glucose_mg_dl"] = (
    28.7 * df["hba1c_baseline"] - 46.7 + np.random.normal(0, 10, len(df))
).round().astype(int).clip(70, 350)
df["postprandial_glucose_mg_dl"] = (
    df["fasting_glucose_mg_dl"] * np.random.uniform(1.3, 1.6, len(df))
).round().astype(int).clip(100, 450)

# ── Final validation ──────────────────────────────────────────
print("\nFinal validation:")
print(f"  NaNs remaining:      {df.isnull().sum().sum()}  (must be 0)")
print(f"  HbA1c < 6.5:         {(df['hba1c_baseline']<6.5).sum()}  (must be 0)")
print(f"  BMI < 22:            {(df['bmi']<22).sum()}  (must be 0)")
print(f"  drug_name NaN:       {df['drug_name'].isna().sum()}  (must be 0)")
print(f"  adverse_event NaN:   {df['adverse_event'].isna().sum()}  (must be 0)")
print(f"  Shape:               {df.shape}")

# ── Save ──────────────────────────────────────────────────────
out = "data/diabetes_dataset_fixed.csv"
df.to_csv(out, index=False)
print(f"\nSaved clean CSV → {out}")
print("Now run:  python train_model.py")