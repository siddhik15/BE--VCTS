# debug_scaler.py
import joblib
import pandas as pd
from ctgan import CTGAN

# Load model
model = CTGAN.load("models/ctgan_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")
columns = joblib.load("models/columns.pkl")

print("="*50)
print("DEBUGGING SCALER")
print("="*50)

print(f"Scaler n_features_in_: {scaler.n_features_in_}")
print(f"Number of numeric columns: {len([c for c in columns if c not in encoders])}")
print(f"Number of categorical columns: {len(encoders)}")
print(f"Total columns: {len(columns)}")

# Get the numeric columns from the model
numeric_cols = [c for c in columns if c not in encoders]
print(f"\nNumeric columns ({len(numeric_cols)}):")
for i, col in enumerate(numeric_cols):
    print(f"  {i}: {col}")

# Check the scaler's min_ and scale_ arrays
print(f"\nScaler min_ length: {len(scaler.min_)}")
print(f"Scaler scale_ length: {len(scaler.scale_)}")
print(f"Scaler data_min_ length: {len(scaler.data_min_)}")

# Generate a sample
print("\nGenerating 5 sample patients...")
samples = model.sample(5)

print(f"\nSample columns in generated data: {list(samples.columns)}")
print(f"Number of columns: {len(samples.columns)}")

# Check which numeric columns are in the sample
present_numeric = [col for col in numeric_cols if col in samples.columns]
print(f"\nPresent numeric columns: {len(present_numeric)}")
print(f"Missing numeric columns: {len([col for col in numeric_cols if col not in samples.columns])}")

# Check the shape of the numeric data
if present_numeric:
    numeric_data = samples[present_numeric].values
    print(f"\nNumeric data shape: {numeric_data.shape}")
    print(f"First row of numeric data (normalized): {numeric_data[0][:5]}...")