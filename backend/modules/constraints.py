"""
constraints.py
FIX: Updated column mapping to match new schema (bmi not bmi_baseline).
"""


def parse_constraints_from_request(request_data):
    """Parse constraints from the frontend request format."""
    params       = request_data.get("population_params", {})
    dataset_type = request_data.get("dataset_type", "readymade")

    if dataset_type != "custom":
        return None

    constraints = {}

    # Map frontend field names → database column names (new schema)
    field_mapping = {
        "age_range":     "age",
        "bmi_range":     "bmi",              # was 'bmi_baseline' / 'BMI'
        "hba1c_range":   "hba1c_baseline",   # was 'baseline_hba1c'
        "glucose_range": "fasting_glucose_mg_dl",
    }

    for frontend_field, db_column in field_mapping.items():
        if frontend_field in params:
            r = params[frontend_field]
            constraints[db_column] = (r["min"], r["max"])

    return constraints


def validate_constraints(constraints):
    """Validate that constraints are logically consistent."""
    errors = []
    if not constraints:
        return errors

    medical_bounds = {
        "age":                   (30,   79),
        "bmi":                   (22.0, 45.0),
        "hba1c_baseline":        (6.5,  12.0),
        "fasting_glucose_mg_dl": (70,   350),
    }

    for col, (min_val, max_val) in constraints.items():
        if min_val >= max_val:
            errors.append(f"{col}: min ({min_val}) must be less than max ({max_val})")
        if min_val < 0:
            errors.append(f"{col}: min cannot be negative")
        if col in medical_bounds:
            med_min, med_max = medical_bounds[col]
            if max_val < med_min:
                errors.append(f"{col}: max {max_val} is below medical minimum {med_min}")
            if min_val > med_max:
                errors.append(f"{col}: min {min_val} is above medical maximum {med_max}")

    return errors