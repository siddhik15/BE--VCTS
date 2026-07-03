# save_default_dataset.py
import os
import pandas as pd
from modules.generate_data import generate_patients

print("="*50)
print("🎯 GENERATING DEFAULT 10,000 PATIENT DATASET")
print("="*50)

# Make sure data directory exists
os.makedirs('data', exist_ok=True)

# Generate 10,000 patients using readymade mode
print("\n📊 Generating patients... (this may take 30-60 seconds)")
df = generate_patients(10000, dataset_type='readymade')

# Save to CSV
output_path = 'data/default_patients.csv'
df.to_csv(output_path, index=False)

print(f"\n✅ SUCCESS! Default dataset saved to: {output_path}")
print(f"📈 Total patients: {len(df)}")
print(f"📋 Columns: {list(df.columns)}")
print("\n🎉 You can now access Feature 2 directly with 10,000 patients!")