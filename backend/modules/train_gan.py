import pandas as pd
import numpy as np
from ctgan import CTGAN
import joblib
import os

def train_gan(data, epochs=500):
    """
    Train CTGAN model on the preprocessed data
    
    Args:
        data: Preprocessed numpy array or DataFrame
        epochs: Number of training epochs
    """
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        columns = ['age', 'gender', 'family_history_diabetes', 'smoking_status',
                  'physical_activity_level', 'diet_quality', 'weight_baseline',
                  'bmi_baseline', 'drug_name', 'dose_mg', 'treatment_group',
                  'baseline_glucose', 'baseline_hba1c', 'glucose_6m', 'glucose_12m',
                  'glucose_36m', 'hba1c_6m', 'hba1c_12m', 'hba1c_36m']
        data = pd.DataFrame(data, columns=columns)
    
    # Identify discrete columns
    discrete_columns = ['gender', 'family_history_diabetes', 'smoking_status',
                       'physical_activity_level', 'diet_quality', 'drug_name',
                       'treatment_group']
    
    # Create and train CTGAN
    model = CTGAN(
        epochs=epochs,
        batch_size=500,
        log_frequency=True,
        verbose=True
    )
    
    print("Training CTGAN model...")
    model.fit(data, discrete_columns)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model.save('models/ctgan_model.pkl')
    print("Model saved to models/ctgan_model.pkl")
    
    return model

if __name__ == "__main__":
    from modules.load_data import load_dataset
    from modules.preprocess_data import preprocess_data
    
    # Load and preprocess data
    df = load_dataset()
    data_scaled, scaler, columns, encoders = preprocess_data(df)
    
    # Train model
    train_gan(data_scaled)
    
    # Save preprocessing objects for later use
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(columns, 'models/columns.pkl')
    print("Preprocessing objects saved")