"""
train_model.py
Trains CTGAN on the cleaned diabetes dataset.

KEY FIX: CTGAN handles its own internal normalization.
         We do NOT apply MinMaxScaler before passing data in.
         We only save the column schema and encoders for post-processing.
"""

import pandas as pd
import numpy as np
import joblib
import os
from ctgan import CTGAN

print("=" * 60)
print("TRAINING CTGAN MODEL FOR SYNTHETIC PATIENT GENERATION")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# 1. Load the dataset
# ─────────────────────────────────────────────────────────────

DATA_PATH = "data\diabetes_dataset_fixed.csv"
df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset: {len(df)} patients, {len(df.columns)} columns")
print(f"Columns: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────
# 2. Clean and validate data
# ─────────────────────────────────────────────────────────────

# Fill NaN values in categorical columns
df["drug_name"] = df["drug_name"].fillna("None")
df["adverse_event"] = df["adverse_event"].fillna("None")
df["responder_status"] = df["responder_status"].fillna("Unknown")

print(f"NaN values after filling:\n{df[['drug_name', 'adverse_event', 'responder_status']].isna().sum()}")

# Convert dose_mg to numeric and handle NaN
df["dose_mg"] = pd.to_numeric(df["dose_mg"], errors="coerce").fillna(0).astype(int)

# Validate data consistency
invalid_drug_dose = ((df["drug_name"] == "None") & (df["dose_mg"] > 0)).sum()
print(f"Invalid records (drug=None but dose>0): {invalid_drug_dose}")

# Fix inconsistent records in training data
df.loc[(df["drug_name"] == "None") & (df["dose_mg"] > 0), "dose_mg"] = 0
df.loc[(df["drug_name"] != "None") & (df["dose_mg"] == 0), "dose_mg"] = 500  # Assign default dose

print(f"Fixed {invalid_drug_dose} inconsistent records")

# ─────────────────────────────────────────────────────────────
# 3. Define column types
# ─────────────────────────────────────────────────────────────

# Explicitly define categorical columns
CATEGORICAL_COLS = [
    "patient_id",
    "gender",
    "treatment_group",
    "drug_name",
    "diet_adherence",
    "medication_adherence",
    "lifestyle_activity",
    "responder_status",
    "adverse_event",
    "dose_mg",          # Discrete doses: 0,10,25,50,100,500,850,1000,1500,2000
]

# Verify all categorical cols exist
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df.columns]

# Convert dose_mg to string so CTGAN treats it as categorical
df["dose_mg"] = df["dose_mg"].astype(str)

# Numeric columns = everything else
NUMERIC_COLS = [c for c in df.columns if c not in CATEGORICAL_COLS]

print(f"\nCategorical columns ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
print(f"Numeric columns ({len(NUMERIC_COLS)}): {NUMERIC_COLS}")

# ─────────────────────────────────────────────────────────────
# 4. Save column schema for generation use
# ─────────────────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)
joblib.dump(df.columns.tolist(),  "models/columns.pkl")
joblib.dump(CATEGORICAL_COLS,     "models/categorical_cols.pkl")
joblib.dump(NUMERIC_COLS,         "models/numeric_cols.pkl")

# Save value ranges for post-generation clipping
ranges = {}
for col in NUMERIC_COLS:
    ranges[col] = (float(df[col].min()), float(df[col].max()))
joblib.dump(ranges, "models/data_ranges.pkl")
print("Saved column schema and data ranges")

# ─────────────────────────────────────────────────────────────
# 5. Train CTGAN
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STARTING CTGAN TRAINING")
print("=" * 60)

model = CTGAN(
    epochs=300,
    batch_size=500,
    generator_dim=(256, 256),
    discriminator_dim=(256, 256),
    generator_lr=0.0002,
    discriminator_lr=0.0002,
    verbose=True
)

model.fit(df, discrete_columns=CATEGORICAL_COLS)

# ─────────────────────────────────────────────────────────────
# 6. Save model
# ─────────────────────────────────────────────────────────────

model.save("models/ctgan_model.pkl")
print("\nModel saved to models/ctgan_model.pkl")

# ─────────────────────────────────────────────────────────────
# 7. Define post-processing validation function
# ─────────────────────────────────────────────────────────────

def validate_and_fix_patient(row):
    """Fix invalid combinations in synthetic data"""
    
    # Convert dose_mg to numeric if it's string
    if isinstance(row['dose_mg'], str):
        try:
            row['dose_mg'] = int(float(row['dose_mg']))
        except:
            row['dose_mg'] = 0
    
    # Fix 1: If drug_name is "None", dose must be 0
    if row['drug_name'] == 'None':
        row['dose_mg'] = 0
        # Treatment group should not have "None" medication (90% confidence)
        if row['treatment_group'] == 'Treatment':
            # 80% chance to assign a drug, 20% stay as None (realistic for non-adherent)
            if np.random.random() < 0.8:
                # Assign most appropriate drug based on patient characteristics
                if 'hba1c_baseline' in row and row['hba1c_baseline'] > 9.0:
                    row['drug_name'] = np.random.choice(['Metformin', 'Empagliflozin'])
                else:
                    row['drug_name'] = np.random.choice(['Metformin', 'Sitagliptin'])
                
                # Assign appropriate dose
                if row['drug_name'] == 'Metformin':
                    row['dose_mg'] = np.random.choice([500, 850, 1000, 1500, 2000])
                elif row['drug_name'] == 'Sitagliptin':
                    row['dose_mg'] = np.random.choice([25, 50, 100])
                else:  # Empagliflozin
                    row['dose_mg'] = np.random.choice([10, 25])
    
    # Fix 2: If dose_mg > 0, drug_name cannot be "None"
    if row['dose_mg'] > 0 and row['drug_name'] == 'None':
        # Assign most common drug for that dose range
        if row['dose_mg'] <= 100:
            row['drug_name'] = 'Sitagliptin'
        elif row['dose_mg'] <= 1000:
            row['drug_name'] = 'Metformin'
        else:
            row['drug_name'] = 'Metformin'
    
    # Fix 3: Treatment group should have medication (mostly)
    if row['treatment_group'] == 'Treatment' and row['drug_name'] == 'None':
        # 85% chance to assign a drug (some patients are non-adherent)
        if np.random.random() < 0.85:
            row['drug_name'] = np.random.choice(['Metformin', 'Sitagliptin', 'Empagliflozin'])
            if row['drug_name'] == 'Metformin':
                row['dose_mg'] = np.random.choice([500, 850, 1000, 1500, 2000])
            elif row['drug_name'] == 'Sitagliptin':
                row['dose_mg'] = np.random.choice([25, 50, 100])
            else:
                row['dose_mg'] = np.random.choice([10, 25])
    
    # Fix 4: Control group should have no medication (95% confidence)
    if row['treatment_group'] == 'Control' and row['drug_name'] != 'None':
        if np.random.random() < 0.95:
            row['drug_name'] = 'None'
            row['dose_mg'] = 0
    
    # Fix 5: Ensure dose_mg is appropriate for drug type
    if row['drug_name'] == 'Metformin' and row['dose_mg'] not in [0, 500, 850, 1000, 1500, 2000]:
        row['dose_mg'] = np.random.choice([500, 850, 1000, 1500, 2000])
    
    if row['drug_name'] == 'Sitagliptin' and row['dose_mg'] not in [0, 25, 50, 100]:
        row['dose_mg'] = np.random.choice([25, 50, 100])
    
    if row['drug_name'] == 'Empagliflozin' and row['dose_mg'] not in [0, 10, 25]:
        row['dose_mg'] = np.random.choice([10, 25])
    
    return row

def generate_validated_samples(model, num_samples, batch_size=1000):
    """Generate and validate synthetic samples"""
    all_samples = []
    
    print(f"\nGenerating {num_samples} synthetic patients in batches of {batch_size}...")
    
    for i in range(0, num_samples, batch_size):
        batch_size_actual = min(batch_size, num_samples - i)
        batch = model.sample(batch_size_actual)
        
        # Apply validation to each row
        batch = batch.apply(validate_and_fix_patient, axis=1)
        
        # Convert dose_mg back to numeric
        if "dose_mg" in batch.columns:
            batch["dose_mg"] = pd.to_numeric(batch["dose_mg"], errors="coerce").fillna(0).astype(int)
        
        all_samples.append(batch)
        
        # Progress indicator
        progress = min(i + batch_size_actual, num_samples)
        print(f"  Generated {progress}/{num_samples} patients ({progress/num_samples*100:.1f}%)", end="\r")
    
    print()  # New line after progress
    return pd.concat(all_samples, ignore_index=True)

# ─────────────────────────────────────────────────────────────
# 8. Quick generation test with validation
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("GENERATING TEST SAMPLES WITH VALIDATION")
print("=" * 60)

# Generate 10 test samples
samples = model.sample(10)
print("\nRaw samples (before validation):")
display_cols = ["patient_id", "age", "hba1c_baseline", "drug_name", "dose_mg", 
                "treatment_group", "medication_adherence", "responder_status"]
available = [c for c in display_cols if c in samples.columns]
print(samples[available].to_string())

# Apply validation
print("\nValidated samples (after fixes):")
samples_validated = samples.apply(validate_and_fix_patient, axis=1)

# Convert dose_mg back to int
if "dose_mg" in samples_validated.columns:
    samples_validated["dose_mg"] = pd.to_numeric(samples_validated["dose_mg"], errors="coerce").fillna(0).astype(int)

print(samples_validated[available].to_string())

# ─────────────────────────────────────────────────────────────
# 9. Generate full synthetic dataset with validation
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("GENERATING FULL SYNTHETIC DATASET")
print("=" * 60)

# CHANGE THIS NUMBER TO CONTROL HOW MANY PATIENTS TO GENERATE
SYNTHETIC_SIZE = 10000  # Generate 10,000 synthetic patients

print(f"\nTarget: {SYNTHETIC_SIZE} synthetic patients")

synthetic_patients = generate_validated_samples(model, SYNTHETIC_SIZE)

# Final cleanup
if "dose_mg" in synthetic_patients.columns:
    synthetic_patients["dose_mg"] = pd.to_numeric(synthetic_patients["dose_mg"], errors="coerce").fillna(0).astype(int)

# Generate unique patient IDs (P00001 to P10000)
num_patients = len(synthetic_patients)
synthetic_patients["patient_id"] = [f"P{str(i+1).zfill(5)}" for i in range(num_patients)]

# Save to CSV
output_path = "synthetic_patients_10000.csv"
synthetic_patients.to_csv(output_path, index=False)
print(f"\n✅ Synthetic dataset saved to {output_path}")
print(f"Generated {len(synthetic_patients)} synthetic patients")

# ─────────────────────────────────────────────────────────────
# 10. Validation Report
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("VALIDATION REPORT")
print("=" * 60)

# Check for invalid combinations
invalid_combos = (
    ((synthetic_patients["drug_name"] == "None") & (synthetic_patients["dose_mg"] > 0)).sum()
)
print(f"Invalid records (drug=None but dose>0): {invalid_combos} ✅" if invalid_combos == 0 else f"❌ Found {invalid_combos} invalid records")

# Check distribution by treatment group
print(f"\n📊 Treatment group distribution:")
treatment_counts = synthetic_patients["treatment_group"].value_counts()
for group, count in treatment_counts.items():
    print(f"  {group}: {count} ({count/len(synthetic_patients)*100:.1f}%)")

# Check distribution by drug
print(f"\n💊 Drug distribution:")
drug_counts = synthetic_patients["drug_name"].value_counts()
for drug, count in drug_counts.items():
    print(f"  {drug}: {count} ({count/len(synthetic_patients)*100:.1f}%)")

# Check responder distribution
print(f"\n🎯 Responder distribution:")
responder_counts = synthetic_patients["responder_status"].value_counts()
for status, count in responder_counts.items():
    print(f"  {status}: {count} ({count/len(synthetic_patients)*100:.1f}%)")

# Check adverse event distribution
print(f"\n⚠️ Adverse event distribution:")
ae_counts = synthetic_patients["adverse_event"].value_counts()
for ae, count in ae_counts.items():
    print(f"  {ae}: {count} ({count/len(synthetic_patients)*100:.1f}%)")

# HbA1c ranges
print(f"\n📈 HbA1c ranges:")
print(f"  Baseline: {synthetic_patients['hba1c_baseline'].min():.1f}% - {synthetic_patients['hba1c_baseline'].max():.1f}%")
print(f"  3 months: {synthetic_patients['hba1c_3_months'].min():.1f}% - {synthetic_patients['hba1c_3_months'].max():.1f}%")
print(f"  6 months: {synthetic_patients['hba1c_6_months'].min():.1f}% - {synthetic_patients['hba1c_6_months'].max():.1f}%")
print(f"  12 months: {synthetic_patients['hba1c_12_months'].min():.1f}% - {synthetic_patients['hba1c_12_months'].max():.1f}%")
print(f"  24 months: {synthetic_patients['hba1c_24_months'].min():.1f}% - {synthetic_patients['hba1c_24_months'].max():.1f}%")
print(f"  36 months: {synthetic_patients['hba1c_36_months'].min():.1f}% - {synthetic_patients['hba1c_36_months'].max():.1f}%")

# Age distribution
print(f"\n👤 Age range: {synthetic_patients['age'].min():.0f} - {synthetic_patients['age'].max():.0f} years")
print(f"  Mean age: {synthetic_patients['age'].mean():.1f} years")

# BMI distribution
if 'bmi' in synthetic_patients.columns:
    print(f"\n⚖️ BMI range: {synthetic_patients['bmi'].min():.1f} - {synthetic_patients['bmi'].max():.1f}")
    print(f"  Mean BMI: {synthetic_patients['bmi'].mean():.1f}")

# Save summary to file
summary_path = "synthetic_summary.txt"
with open(summary_path, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("SYNTHETIC DATASET SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total patients: {len(synthetic_patients)}\n")
    f.write(f"Total columns: {len(synthetic_patients.columns)}\n\n")
    f.write("Treatment group distribution:\n")
    for group, count in treatment_counts.items():
        f.write(f"  {group}: {count} ({count/len(synthetic_patients)*100:.1f}%)\n")
    f.write("\nDrug distribution:\n")
    for drug, count in drug_counts.items():
        f.write(f"  {drug}: {count} ({count/len(synthetic_patients)*100:.1f}%)\n")
    f.write("\nResponder distribution:\n")
    for status, count in responder_counts.items():
        f.write(f"  {status}: {count} ({count/len(synthetic_patients)*100:.1f}%)\n")

print(f"\n📄 Summary saved to {summary_path}")

print("\n" + "=" * 60)
print("✅ CTGAN TRAINING AND SYNTHETIC GENERATION COMPLETE")
print("=" * 60)
print(f"\nOutput files:")
print(f"  - models/ctgan_model.pkl (trained model)")
print(f"  - {output_path} (10,000 synthetic patients)")
print(f"  - {summary_path} (dataset summary)")