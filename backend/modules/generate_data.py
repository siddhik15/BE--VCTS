"""
generate_data.py
Diabetes patient generator with dual-mode support.
- readymade: Rule-based synthetic generation (ADA 2025 guidelines)
- ctgan: AI-generated patients using trained CTGAN model
- custom: Rule-based with constraints

FIXES:
- Random mixing of treatment/control groups (not sequential)
- Clean separation: Control = No drug, Treatment = Active drugs only
- 70% Treatment / 30% Control distribution
- Only 3 responder types: Responder, Partial, Non-Responder
"""

import numpy as np
import pandas as pd
import random
import warnings
import os

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CTGAN Model Loader (Optional)
# ─────────────────────────────────────────────────────────────

class CTGANDataGenerator:
    """Wrapper for CTGAN model to generate synthetic patients"""
    
    _model = None
    _columns = None
    _categorical_cols = None
    _numeric_cols = None
    _ranges = None
    _valid_doses = {
        "Metformin":     [500, 850, 1000, 1500, 2000],
        "Sitagliptin":   [25, 50, 100],
        "Empagliflozin": [10, 25],
        "None":          [0],
    }
    _medical_bounds = {
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
    }
    
    @classmethod
    def load_model(cls, model_path="models/ctgan_model.pkl"):
        """Load CTGAN model and schemas"""
        if cls._model is None:
            try:
                from ctgan import CTGAN
                import joblib
                
                if os.path.exists(model_path):
                    cls._model = CTGAN.load(model_path)
                    cls._columns = joblib.load("models/columns.pkl")
                    cls._categorical_cols = joblib.load("models/categorical_cols.pkl")
                    cls._numeric_cols = joblib.load("models/numeric_cols.pkl")
                    cls._ranges = joblib.load("models/data_ranges.pkl")
                    print("✅ CTGAN model loaded successfully")
                else:
                    print(f"⚠️ Model not found at {model_path}")
                    return None
            except Exception as e:
                print(f"⚠️ Error loading CTGAN model: {e}")
                return None
        return cls._model
    
    @classmethod
    def generate(cls, n_patients=1000, validate=True):
        """Generate synthetic patients using CTGAN"""
        model = cls.load_model()
        if model is None:
            return None
        
        print(f"Generating {n_patients} patients with CTGAN...")
        samples = model.sample(n_patients)
        
        if validate:
            samples = cls._validate_samples(samples)
        
        # Generate unique patient IDs
        samples["patient_id"] = [f"CTGAN_{str(i+1).zfill(5)}" for i in range(len(samples))]
        
        return samples
    
    @classmethod
    def _validate_samples(cls, df):
        """Apply business rules to fix invalid combinations"""
        df = df.copy()
        
        # Fix 1: drug=None must have dose=0
        if "drug_name" in df.columns and "dose_mg" in df.columns:
            df["dose_mg"] = pd.to_numeric(df["dose_mg"], errors="coerce").fillna(0).astype(int)
            df.loc[(df["drug_name"] == "None") & (df["dose_mg"] > 0), "dose_mg"] = 0
        
        # Fix 2: dose>0 must have actual drug
        df.loc[(df["dose_mg"] > 0) & (df["drug_name"] == "None"), "drug_name"] = "Metformin"
        
        # Fix 3: Validate drug-dose combinations
        if "drug_name" in df.columns and "dose_mg" in df.columns:
            for idx, row in df.iterrows():
                drug = str(row.get("drug_name", "None"))
                dose = int(row.get("dose_mg", 0))
                if drug in cls._valid_doses:
                    valid = cls._valid_doses[drug]
                    if dose not in valid:
                        df.at[idx, "dose_mg"] = min(valid, key=lambda x: abs(x - dose))
                else:
                    df.at[idx, "dose_mg"] = 0
        
        # Fix 4: Clip numeric columns to medical bounds
        for col, (lo, hi) in cls._medical_bounds.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].clip(lo, hi)
        
        # Fix 5: ENFORCE CLEAN SEPARATION - NO CROSSOVER
        # Treatment group MUST have active drug, Control group MUST have None
        if "treatment_group" in df.columns and "drug_name" in df.columns:
            # Treatment group: ensure all have active drugs (no "None")
            treatment_mask = df["treatment_group"] == "Treatment"
            if treatment_mask.any():
                none_in_treatment = treatment_mask & (df["drug_name"] == "None")
                n_fix = none_in_treatment.sum()
                if n_fix > 0:
                    drugs_to_assign = np.random.choice(
                        ["Metformin", "Sitagliptin", "Empagliflozin"], 
                        size=n_fix,
                        p=[0.55, 0.25, 0.20]
                    )
                    df.loc[none_in_treatment, "drug_name"] = drugs_to_assign
                    for idx in df[none_in_treatment].index:
                        drug = df.at[idx, "drug_name"]
                        df.at[idx, "dose_mg"] = np.random.choice(cls._valid_doses[drug])
            
            # Control group: ensure all have "None" drug
            control_mask = df["treatment_group"] == "Control"
            if control_mask.any():
                df.loc[control_mask, "drug_name"] = "None"
                df.loc[control_mask, "dose_mg"] = 0
        
        return df
    
    @classmethod
    def recalculate_responder_status(cls, df):
        """Recalculate responder status - ONLY 3 TYPES"""
        if "hba1c_baseline" in df.columns and "hba1c_12_months" in df.columns:
            reduction = df["hba1c_baseline"] - df["hba1c_12_months"]
            conditions = [
                reduction >= 1.0,   # 1.0% or more reduction
                reduction >= 0.0,   # 0% to 0.99% reduction
            ]
            choices = [
                "Responder",
                "Partial",
            ]
            df["responder_status"] = np.select(conditions, choices, default="Non-Responder")
        return df
    
    @classmethod
    def recalculate_risk_probabilities(cls, df):
        """Recalculate diabetes risk probabilities from generated HbA1c values"""
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
# Rule-Based Generator
# ─────────────────────────────────────────────────────────────

# Drug-dose efficacy (ADA 2025, UKPDS, EMPA-REG, TECOS trials)
DRUG_DOSE_EFFICACY = {
    ("Metformin",      500):  (0.70, 0.22),
    ("Metformin",      850):  (1.00, 0.25),
    ("Metformin",     1000):  (1.15, 0.27),
    ("Metformin",     1500):  (1.35, 0.28),
    ("Metformin",     2000):  (1.50, 0.30),
    ("Sitagliptin",     25):  (0.42, 0.15),
    ("Sitagliptin",     50):  (0.62, 0.18),
    ("Sitagliptin",    100):  (0.82, 0.22),
    ("Empagliflozin",   10):  (0.68, 0.20),
    ("Empagliflozin",   25):  (0.88, 0.22),
    ("None",             0):  (0.00, 0.10),
}

# Adherence multipliers
MED_ADHERENCE_MULT  = {"Poor": 0.35, "Moderate": 0.65, "Good": 1.00}
DIET_MULT           = {"Poor": 0.72, "Moderate": 0.86, "Good": 1.00}
LIFESTYLE_MULT      = {"Sedentary": 0.78, "Moderate": 0.92, "Active": 1.10}

# Control group lifestyle + diet effect without drug
CONTROL_LIFESTYLE_DELTA = {"Sedentary": -0.20, "Moderate": 0.10, "Active": 0.38}
CONTROL_DIET_DELTA      = {"Poor": -0.12, "Moderate": 0.05, "Good": 0.18}

# Adverse event profiles
ADVERSE_EVENT_PROFILE = {
    "Metformin":      {"GI Issues": 0.20, "Headache": 0.03, "None": 0.77},
    "Sitagliptin":    {"Headache": 0.08, "GI Issues": 0.04, "None": 0.88},
    "Empagliflozin":  {"UTI": 0.08, "Hypoglycemia": 0.05, "Mild": 0.04, "None": 0.83},
    "None":           {"None": 1.00},
}


class DiabetesDataGenerator:

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.GENETIC = {"Super_Responder": 0.05, "Normal": 0.80, "Poor_Responder": 0.15}

    def _sample(self, d):
        return np.random.choice(list(d.keys()), p=list(d.values()))

    def _clip(self, x, a, b):
        return max(a, min(b, x))

    # ── demographics ─────────────────────────────────────────

    def _demographics(self):
        age = int(self._clip(np.random.gamma(7, 6) + 35, 30, 79))
        gender = np.random.choice(["Male", "Female"], p=[0.48, 0.52])

        bmi_mean = 30.0 + np.random.choice([0, 1.5, 1.0, -0.5], p=[0.64, 0.13, 0.16, 0.07])
        bmi = round(self._clip(np.random.normal(bmi_mean, 5), 22.0, 45.0), 1)

        med_adh = self._sample({"Poor": 0.20, "Moderate": 0.35, "Good": 0.45})
        diet_adh = self._sample({"Poor": 0.20, "Moderate": 0.40, "Good": 0.40})
        lifestyle = self._sample({"Sedentary": 0.32, "Moderate": 0.38, "Active": 0.30})

        return dict(age=age, gender=gender, bmi=bmi,
                    medication_adherence=med_adh,
                    diet_adherence=diet_adh,
                    lifestyle_activity=lifestyle)

    # ── baseline HbA1c ───────────────────────────────────────

    def _baseline_hba1c(self, age, bmi, diet, lifestyle):
        """T2D: 6.5–12.0% (ADA). Higher BMI/age/poor lifestyle → higher."""
        base = 6.5 + (age - 40) * 0.025 + (bmi - 27) * 0.045
        if lifestyle == "Sedentary": base += 0.30
        if lifestyle == "Active":    base -= 0.20
        if diet == "Poor":           base += 0.20
        if diet == "Good":           base -= 0.10
        base += np.random.normal(0, 0.5)
        genetic = self._sample(self.GENETIC)
        if genetic == "Super_Responder": base -= 0.30
        if genetic == "Poor_Responder":  base += 0.40
        return round(self._clip(base, 6.5, 12.0), 1), genetic

    # ── treatment assignment ─────────────────────────────────

    def _treatment(self, hba1c):
        """ADA algorithm: higher HbA1c → more aggressive dosing."""
        if hba1c < 7.0:
            drug = "Metformin"
            dose = int(np.random.choice([500, 850]))
        elif hba1c < 7.5:
            drug = "Metformin"
            dose = int(np.random.choice([850, 1000]))
        elif hba1c < 8.0:
            drug = str(np.random.choice(["Metformin", "Empagliflozin", "Sitagliptin"],
                                    p=[0.60, 0.20, 0.20]))
            if drug == "Metformin":
                dose = int(np.random.choice([1000, 1500]))
            elif drug == "Empagliflozin":
                dose = 10
            else:
                dose = 50
        elif hba1c < 9.0:
            drug = str(np.random.choice(["Metformin", "Empagliflozin", "Sitagliptin"],
                                    p=[0.50, 0.25, 0.25]))
            if drug == "Metformin":
                dose = int(np.random.choice([1500, 2000]))
            elif drug == "Empagliflozin":
                dose = 25
            else:
                dose = 100
        else:
            drug = str(np.random.choice(["Metformin", "Empagliflozin", "Sitagliptin"],
                                    p=[0.40, 0.35, 0.25]))
            if drug == "Metformin":
                dose = 2000
            elif drug == "Empagliflozin":
                dose = 25
            else:
                dose = 100
        return drug, dose
    
    # ── HbA1c at follow-up ───────────────────────────────────

    def _hba1c_at_month(self, baseline, drug, dose, months,
                         med_adh, diet, lifestyle, group):
        """
        Treatment: reduction = efficacy × adherence × diet × lifestyle
                               × severity_boost × time_curve
        Control:   change driven by lifestyle + diet only (no drug)
        """
        if group == "Control":
            life_delta = CONTROL_LIFESTYLE_DELTA.get(lifestyle, 0.0)
            diet_delta = CONTROL_DIET_DELTA.get(diet, 0.0)
            time_factor = min(1.0, months / 12.0)
            change = (life_delta + diet_delta) * time_factor
            result = baseline - change + np.random.normal(0, 0.12)
        else:
            key = (drug, int(dose))
            mean_red, std_red = DRUG_DOSE_EFFICACY.get(key, (0.80, 0.25))

            m_mult = MED_ADHERENCE_MULT.get(med_adh, 0.65)
            d_mult = DIET_MULT.get(diet, 0.86)
            l_mult = LIFESTYLE_MULT.get(lifestyle, 0.92)

            # Higher baseline → bigger absolute reduction
            severity = 1.0 + max(0.0, (baseline - 8.0) * 0.05)

            # Time-response curve
            if months <= 3:
                t = months / 6.0
            elif months <= 6:
                t = 0.50 + (months - 3) / 12.0
            elif months <= 12:
                t = 1.00 + (months - 6) / 30.0
            elif months <= 24:
                t = 1.20 + (months - 12) / 48.0
            else:
                t = 1.45 + (months - 24) / 96.0

            reduction = mean_red * m_mult * d_mult * l_mult * severity * t
            result = baseline - reduction + np.random.normal(0, std_red * 0.45)

        return round(float(self._clip(result, 4.5, 14.0)), 1)

    # ── responder status - ONLY 3 TYPES ───────────────────────

    def _responder_status(self, baseline, hba1c_12m):
        """Only 3 categories: Responder, Partial, Non-Responder"""
        r = baseline - hba1c_12m
        if r >= 1.0:   # 1.0% or more reduction
            return "Responder"
        elif r >= 0.0:  # 0% to 0.99% reduction (no change or slight improvement)
            return "Partial"
        else:           # Negative reduction (worsening)
            return "Non-Responder"

    # ── adverse event ────────────────────────────────────────

    def _adverse_event(self, drug):
        profile = ADVERSE_EVENT_PROFILE.get(drug, {"None": 1.0})
        return np.random.choice(list(profile.keys()), p=list(profile.values()))

    # ── lab values ───────────────────────────────────────────

    def _labs(self, hba1c, age, bmi):
        fasting = int(self._clip(28.7 * hba1c - 46.7 + np.random.normal(0, 10), 70, 350))
        pp = int(self._clip(fasting * np.random.uniform(1.3, 1.6), 100, 450))
        chol = int(self._clip(160 + (bmi - 25) * 2 + (age - 40) * 1.5 + np.random.normal(0, 20), 100, 350))
        trig = int(self._clip(120 + (bmi - 25) * 5 + (hba1c - 6) * 18 + np.random.normal(0, 30), 50, 500))
        sys_bp = int(self._clip(115 + (age - 40) * 0.5 + (bmi - 25) * 1.2 + np.random.normal(0, 8), 90, 180))
        dia_bp = int(self._clip(70 + (age - 40) * 0.1 + (bmi - 25) * 0.3 + np.random.normal(0, 5), 55, 105))
        return dict(fasting_glucose_mg_dl=fasting, postprandial_glucose_mg_dl=pp,
                    cholesterol_mg_dl=chol, triglycerides_mg_dl=trig,
                    systolic_bp=sys_bp, diastolic_bp=dia_bp)

    def _risk_prob(self, hba1c):
        """Logistic: risk rises sharply above 7.5% HbA1c."""
        return round(float(self._clip(1 / (1 + np.exp(-(hba1c - 7.5) / 1.2)), 0.05, 0.95)), 3)

    # ── main generator (RANDOM MIXED) ─────────────────────────

    def generate_dataset(self, n_patients=1000):
        """
        Generate dataset with:
        - 70% Treatment group (active drugs)
        - 30% Control group (no drugs)
        - RANDOMLY MIXED order (not sequential)
        - Clean separation: Control = None, Treatment = active drug only
        """
        patients = []
        
        # Calculate group sizes
        n_treatment = int(n_patients * 0.7)
        n_control = n_patients - n_treatment
        
        # Create shuffled group assignments (random mixing)
        group_assignments = ["Control"] * n_control + ["Treatment"] * n_treatment
        random.shuffle(group_assignments)
        
        for i in range(n_patients):
            pid = f"P{str(i + 1).zfill(5)}"
            demo = self._demographics()
            baseline, genetic = self._baseline_hba1c(
                demo["age"], demo["bmi"],
                demo["diet_adherence"], demo["lifestyle_activity"]
            )
            
            # Get randomly assigned group
            group = group_assignments[i]
            
            if group == "Control":
                drug, dose = "None", 0
            else:
                drug, dose = self._treatment(baseline)

            h3 = self._hba1c_at_month(baseline, drug, dose, 3,
                    demo["medication_adherence"], demo["diet_adherence"],
                    demo["lifestyle_activity"], group)
            h6 = self._hba1c_at_month(baseline, drug, dose, 6,
                    demo["medication_adherence"], demo["diet_adherence"],
                    demo["lifestyle_activity"], group)
            h12 = self._hba1c_at_month(baseline, drug, dose, 12,
                    demo["medication_adherence"], demo["diet_adherence"],
                    demo["lifestyle_activity"], group)
            h24 = self._hba1c_at_month(baseline, drug, dose, 24,
                    demo["medication_adherence"], demo["diet_adherence"],
                    demo["lifestyle_activity"], group)
            h36 = self._hba1c_at_month(baseline, drug, dose, 36,
                    demo["medication_adherence"], demo["diet_adherence"],
                    demo["lifestyle_activity"], group)

            labs = self._labs(baseline, demo["age"], demo["bmi"])
            responder = self._responder_status(baseline, h12)
            ae = self._adverse_event(drug)

            patients.append({
                "patient_id": pid,
                "age": demo["age"], "gender": demo["gender"], "bmi": demo["bmi"],
                "treatment_group": group, "drug_name": drug, "dose_mg": dose,
                "diet_adherence": demo["diet_adherence"],
                "medication_adherence": demo["medication_adherence"],
                "lifestyle_activity": demo["lifestyle_activity"],
                "hba1c_baseline": baseline,
                "hba1c_3_months": h3, "hba1c_6_months": h6,
                "hba1c_12_months": h12, "hba1c_24_months": h24, "hba1c_36_months": h36,
                "responder_status": responder,
                "adverse_event": ae,
                "diabetes_risk_probability_3m": self._risk_prob(h3),
                "diabetes_risk_probability_6m": self._risk_prob(h6),
                "diabetes_risk_probability_12m": self._risk_prob(h12),
                "diabetes_risk_probability_24m": self._risk_prob(h24),
                "diabetes_risk_probability_36m": self._risk_prob(h36),
                **labs,
            })

        # Final shuffle for extra randomness
        random.shuffle(patients)
        return pd.DataFrame(patients)
    
    # ── custom generator (RANDOM MIXED with constraints) ─────

    def generate_custom_dataset(self, n_patients=1000, constraints=None):
        """
        Generate dataset with custom constraints and random mixing.
        """
        patients = []
        
        # Calculate group sizes (70% treatment, 30% control)
        n_treatment = int(n_patients * 0.7)
        n_control = n_patients - n_treatment
        
        # Create shuffled group assignments (random mixing)
        group_assignments = ["Control"] * n_control + ["Treatment"] * n_treatment
        random.shuffle(group_assignments)
        
        # Set up constraint ranges
        age_min = constraints.get("age_min", 30) if constraints else 30
        age_max = constraints.get("age_max", 79) if constraints else 79
        bmi_min = constraints.get("bmi_min", 22.0) if constraints else 22.0
        bmi_max = constraints.get("bmi_max", 45.0) if constraints else 45.0
        hba1c_min = constraints.get("hba1c_min", 6.5) if constraints else 6.5
        hba1c_max = constraints.get("hba1c_max", 12.0) if constraints else 12.0
        glucose_min = constraints.get("glucose_min", 70) if constraints else 70
        glucose_max = constraints.get("glucose_max", 350) if constraints else 350
        
        # Validate constraints against medical guidelines
        age_min = max(30, min(79, age_min))
        age_max = min(79, max(30, age_max))
        bmi_min = max(22.0, min(45.0, bmi_min))
        bmi_max = min(45.0, max(22.0, bmi_max))
        hba1c_min = max(6.5, min(12.0, hba1c_min))
        hba1c_max = min(12.0, max(6.5, hba1c_max))
        glucose_min = max(70, min(350, glucose_min))
        glucose_max = min(350, max(70, glucose_max))
        
        generated = 0
        max_attempts = n_patients * 10
        
        while generated < n_patients and len(patients) < max_attempts:
            # Generate patient with random values within constraints
            age = np.random.randint(age_min, age_max + 1)
            bmi = round(np.random.uniform(bmi_min, bmi_max), 1)
            gender = np.random.choice(["Male", "Female"], p=[0.48, 0.52])
            
            base_hba1c = np.random.uniform(hba1c_min, hba1c_max)
            
            if base_hba1c < hba1c_min or base_hba1c > hba1c_max:
                continue
            
            # Assign adherence levels
            med_adh = np.random.choice(["Poor", "Moderate", "Good"], p=[0.20, 0.35, 0.45])
            diet_adh = np.random.choice(["Poor", "Moderate", "Good"], p=[0.20, 0.40, 0.40])
            lifestyle = np.random.choice(["Sedentary", "Moderate", "Active"], p=[0.32, 0.38, 0.30])
            
            # Genetic factor
            genetic = np.random.choice(["Super_Responder", "Normal", "Poor_Responder"], 
                                       p=[0.05, 0.80, 0.15])
            if genetic == "Super_Responder":
                base_hba1c -= 0.30
            elif genetic == "Poor_Responder":
                base_hba1c += 0.40
            
            baseline = round(self._clip(base_hba1c, 6.5, 12.0), 1)
            
            # Get randomly assigned group
            group = group_assignments[generated]
            
            if group == "Control":
                drug, dose = "None", 0
            else:
                drug, dose = self._treatment(baseline)
            
            # Calculate follow-up HbA1c
            h3 = self._hba1c_at_month(baseline, drug, dose, 3, med_adh, diet_adh, lifestyle, group)
            h6 = self._hba1c_at_month(baseline, drug, dose, 6, med_adh, diet_adh, lifestyle, group)
            h12 = self._hba1c_at_month(baseline, drug, dose, 12, med_adh, diet_adh, lifestyle, group)
            h24 = self._hba1c_at_month(baseline, drug, dose, 24, med_adh, diet_adh, lifestyle, group)
            h36 = self._hba1c_at_month(baseline, drug, dose, 36, med_adh, diet_adh, lifestyle, group)
            
            # Calculate labs
            fasting = int(self._clip(28.7 * baseline - 46.7 + np.random.normal(0, 10), 70, 350))
            
            if fasting < glucose_min or fasting > glucose_max:
                continue
            
            pp = int(self._clip(fasting * np.random.uniform(1.3, 1.6), 100, 450))
            chol = int(self._clip(160 + (bmi - 25) * 2 + (age - 40) * 1.5 + np.random.normal(0, 20), 100, 350))
            trig = int(self._clip(120 + (bmi - 25) * 5 + (baseline - 6) * 18 + np.random.normal(0, 30), 50, 500))
            sys_bp = int(self._clip(115 + (age - 40) * 0.5 + (bmi - 25) * 1.2 + np.random.normal(0, 8), 90, 180))
            dia_bp = int(self._clip(70 + (age - 40) * 0.1 + (bmi - 25) * 0.3 + np.random.normal(0, 5), 55, 105))
            
            responder = self._responder_status(baseline, h12)
            ae = self._adverse_event(drug)
            
            patients.append({
                "patient_id": f"P{str(generated + 1).zfill(5)}",
                "age": age, "gender": gender, "bmi": bmi,
                "treatment_group": group, "drug_name": drug, "dose_mg": dose,
                "diet_adherence": diet_adh,
                "medication_adherence": med_adh,
                "lifestyle_activity": lifestyle,
                "hba1c_baseline": baseline,
                "hba1c_3_months": h3, "hba1c_6_months": h6,
                "hba1c_12_months": h12, "hba1c_24_months": h24, "hba1c_36_months": h36,
                "responder_status": responder,
                "adverse_event": ae,
                "diabetes_risk_probability_3m": self._risk_prob(h3),
                "diabetes_risk_probability_6m": self._risk_prob(h6),
                "diabetes_risk_probability_12m": self._risk_prob(h12),
                "diabetes_risk_probability_24m": self._risk_prob(h24),
                "diabetes_risk_probability_36m": self._risk_prob(h36),
                "fasting_glucose_mg_dl": fasting,
                "postprandial_glucose_mg_dl": pp,
                "cholesterol_mg_dl": chol,
                "triglycerides_mg_dl": trig,
                "systolic_bp": sys_bp,
                "diastolic_bp": dia_bp,
            })
            generated += 1
        
        # Final shuffle for extra randomness
        random.shuffle(patients)
        return pd.DataFrame(patients)


# ─────────────────────────────────────────────────────────────
# MAIN GENERATION FUNCTION
# ─────────────────────────────────────────────────────────────

def generate_patients(n=1000, constraints=None, dataset_type="readymade"):
    """
    Main entry point for patient generation.
    
    Args:
        n: Number of patients to generate
        constraints: Dict of constraints for filtering (for custom mode)
        dataset_type: "readymade" (rule-based), "ctgan" (AI-generated), or "custom"
    
    Returns:
        DataFrame of generated patients
    """
    
    if dataset_type == "ctgan":
        df = CTGANDataGenerator.generate(n_patients=n, validate=True)
        if df is not None:
            df = CTGANDataGenerator.recalculate_responder_status(df)
            df = CTGANDataGenerator.recalculate_risk_probabilities(df)
            return df
        else:
            print("⚠️ CTGAN generation failed, falling back to rule-based generator")
            dataset_type = "readymade"
    
    if dataset_type == "readymade":
        generator = DiabetesDataGenerator()
        return generator.generate_dataset(n)
    
    elif dataset_type == "custom":
        generator = DiabetesDataGenerator()
        if constraints:
            custom_constraints = {}
            if "age_range" in constraints:
                custom_constraints["age_min"] = constraints["age_range"]["min"]
                custom_constraints["age_max"] = constraints["age_range"]["max"]
            if "bmi_range" in constraints:
                custom_constraints["bmi_min"] = constraints["bmi_range"]["min"]
                custom_constraints["bmi_max"] = constraints["bmi_range"]["max"]
            if "hba1c_range" in constraints:
                custom_constraints["hba1c_min"] = constraints["hba1c_range"]["min"]
                custom_constraints["hba1c_max"] = constraints["hba1c_range"]["max"]
            if "glucose_range" in constraints:
                custom_constraints["glucose_min"] = constraints["glucose_range"]["min"]
                custom_constraints["glucose_max"] = constraints["glucose_range"]["max"]
            df = generator.generate_custom_dataset(n, constraints=custom_constraints)
        else:
            df = generator.generate_custom_dataset(n)
        return df
    
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")


# ─────────────────────────────────────────────────────────────
# COMMAND LINE INTERFACE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic diabetes patient data")
    parser.add_argument("--n", type=int, default=1000, help="Number of patients to generate")
    parser.add_argument("--type", choices=["readymade", "ctgan", "custom"], 
                        default="readymade", help="Generation type")
    parser.add_argument("--output", type=str, default="generated_patients.csv", 
                        help="Output file name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DIABETES PATIENT GENERATOR")
    print("=" * 60)
    print(f"Mode: {args.type}")
    print(f"Patients: {args.n}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    df = generate_patients(n=args.n, dataset_type=args.type)
    
    df.to_csv(args.output, index=False)
    print(f"\n✅ Saved {len(df)} patients to {args.output}")
    
    print("\n📊 Sample (first 15 patients):")
    cols = ["patient_id", "age", "bmi", "treatment_group", "drug_name",
            "dose_mg", "hba1c_baseline", "hba1c_12_months", "responder_status"]
    avail = [c for c in cols if c in df.columns]
    print(df[avail].head(15).to_string())
    
    print("\n📈 Summary Statistics:")
    print(f"  Treatment group: {len(df[df['treatment_group'] == 'Treatment'])}")
    print(f"  Control group: {len(df[df['treatment_group'] == 'Control'])}")
    print(f"  Mean age: {df['age'].mean():.1f}")
    print(f"  Mean BMI: {df['bmi'].mean():.1f}")
    print(f"  Mean baseline HbA1c: {df['hba1c_baseline'].mean():.1f}%")
    
    print("\n🎯 Responder distribution (3 types: Responder, Partial, Non-Responder):")
    print(df["responder_status"].value_counts())
    
    print("\n💊 Drug distribution by group:")
    print("\nTreatment group:")
    treatment_df = df[df["treatment_group"] == "Treatment"]
    if len(treatment_df) > 0:
        print(treatment_df["drug_name"].value_counts())
    print("\nControl group:")
    control_df = df[df["treatment_group"] == "Control"]
    if len(control_df) > 0:
        print(control_df["drug_name"].value_counts())
    
    print("\n⚠️ Adverse event distribution:")
    print(df["adverse_event"].value_counts())