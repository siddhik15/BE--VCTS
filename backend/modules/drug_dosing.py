import pandas as pd
import numpy as np

class DiabetesMedications:
    """Medication dosing and efficacy models"""
    
    # Metformin dosing
    METFORMIN_DOSES = [500, 850, 1000, 1500, 2000, 2500]
    METFORMIN_STANDARD_DOSE = 2000  # mg/day maximum
    
    # SGLT2 inhibitors (typical doses)
    SGLT2I_DOSES = {
        'Empagliflozin': [10, 25],
        'Dapagliflozin': [5, 10],
        'Canagliflozin': [100, 300]
    }
    
    # GLP-1 receptor agonists
    GLP1_DOSES = {
        'Liraglutide': [0.6, 1.2, 1.8],
        'Semaglutide': [0.5, 1.0, 2.0],
        'Dulaglutide': [0.75, 1.5, 3.0, 4.5]
    }
    
    # DPP-4 inhibitors
    DPP4I_DOSES = {
        'Sitagliptin': [25, 50, 100],
        'Saxagliptin': [2.5, 5],
        'Linagliptin': [5]
    }
    
    @classmethod
    def validate_dose(cls, drug_name, dose_mg):
        """Check if dose is within approved ranges"""
        # Convert to string and handle None
        if drug_name is None:
            return True, "No drug specified"
        
        # Convert to string if it's a numpy float or other type
        if not isinstance(drug_name, str):
            try:
                drug_name = str(drug_name)
            except:
                return True, f"Invalid drug name type: {type(drug_name)}"
        
        drug_name_lower = drug_name.lower()
        
        # Metformin
        if 'metformin' in drug_name_lower:
            return dose_mg in cls.METFORMIN_DOSES, f"Metformin dose must be one of {cls.METFORMIN_DOSES}mg"
        
        # SGLT2 inhibitors
        for drug, doses in cls.SGLT2I_DOSES.items():
            if drug.lower() in drug_name_lower:
                return dose_mg in doses, f"{drug} dose must be one of {doses}mg"
        
        # GLP-1 agonists
        for drug, doses in cls.GLP1_DOSES.items():
            if drug.lower() in drug_name_lower:
                return dose_mg in doses, f"{drug} dose must be one of {doses}mg"
        
        # DPP-4 inhibitors
        for drug, doses in cls.DPP4I_DOSES.items():
            if drug.lower() in drug_name_lower:
                return dose_mg in doses, f"{drug} dose must be one of {doses}mg"
        
        return True, "Unknown drug, dose not validated"
    
    @classmethod
    def calculate_expected_hba1c_reduction(cls, drug_name, baseline_hba1c, duration_months=6):
        """
        Calculate expected HbA1c reduction based on clinical trial data
        """
        # Convert to string and handle None
        if drug_name is None:
            return np.random.uniform(0.5, 1.0)
        
        if not isinstance(drug_name, str):
            try:
                drug_name = str(drug_name)
            except:
                return np.random.uniform(0.5, 1.0)
        
        drug_name_lower = drug_name.lower()
        baseline = float(baseline_hba1c)
        
        # Expected reductions from clinical trials
        if 'metformin' in drug_name_lower:
            reduction = np.random.uniform(1.0, 1.5)  # 1-1.5% reduction
        elif any(sglt2 in drug_name_lower for sglt2 in ['empagliflozin', 'dapagliflozin', 'canagliflozin']):
            reduction = np.random.uniform(0.8, 1.2)  # SGLT2 inhibitors
        elif any(glp1 in drug_name_lower for glp1 in ['liraglutide', 'semaglutide', 'dulaglutide']):
            reduction = np.random.uniform(1.0, 1.8)  # GLP-1 agonists
        elif any(dpp4 in drug_name_lower for dpp4 in ['sitagliptin', 'saxagliptin', 'linagliptin']):
            reduction = np.random.uniform(0.6, 0.8)  # DPP-4 inhibitors
        else:
            reduction = np.random.uniform(0.5, 1.0)  # Other
        
        # Adjust for baseline (higher baseline = greater reduction)
        baseline_factor = 1.0 + (baseline - 7.0) * 0.1
        reduction *= min(baseline_factor, 1.3)  # Cap at 30% increase
        
        return reduction