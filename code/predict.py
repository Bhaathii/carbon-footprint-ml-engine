import joblib
import numpy as np
import pandas as pd
import os
from recomendation import get_recommendation

# Load trained model and features
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "../model/features.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

# Example: Create sample data for prediction
# Using values from the dataset
sample_data = {
    'trip_days': 6,
    'distance_km': 54.8,
    'occupancy': 27,
    'congestion_factor': 0.91,
    'terrain_factor': 1.15,
    'nightly_kwh_est': 8.44,
    'grid_ef_kgCO2_per_kWh': 0.07,
    'food_emissions_kgCO2': 16.884,
    'waste_emissions_kgCO2': 2.998,
    'plastic_emissions_kgCO2': 0.3,
    'transport_emissions_kgCO2': 0.037,
    'accommodation_elec_kgCO2': 4.805,
    'accommodation_gen_kgCO2': 11.095
}

# Create DataFrame and ensure column order
data_df = pd.DataFrame([sample_data])
data = data_df[feature_columns].values

pred = model.predict(data)[0]

print("\n" + "="*50)
print("PREDICTION RESULT")
print("="*50)
print("Predicted Emission Level:", pred.upper())
print("\nRecommendation:")
print(get_recommendation(pred))
print("="*50)
