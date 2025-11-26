import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib, os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), "../data/tourism_5000_rows.csv")
df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# Create emission level classification based on total_trip_emissions_kgCO2
# Define thresholds for low, medium, high
total_emissions = df['total_trip_emissions_kgCO2']
percentile_33 = total_emissions.quantile(0.33)
percentile_67 = total_emissions.quantile(0.67)

def classify_emission(value):
    if value <= percentile_33:
        return 'low'
    elif value <= percentile_67:
        return 'medium'
    else:
        return 'high'

df['emission_level'] = df['total_trip_emissions_kgCO2'].apply(classify_emission)

print("\nEmission Level Distribution:")
print(df['emission_level'].value_counts())

# Select relevant features for prediction
feature_columns = [
    'trip_days', 'distance_km', 'occupancy', 'congestion_factor', 
    'terrain_factor', 'nightly_kwh_est', 'grid_ef_kgCO2_per_kWh',
    'food_emissions_kgCO2', 'waste_emissions_kgCO2', 
    'plastic_emissions_kgCO2', 'transport_emissions_kgCO2',
    'accommodation_elec_kgCO2', 'accommodation_gen_kgCO2'
]

# Check for missing columns
missing_cols = [col for col in feature_columns if col not in df.columns]
if missing_cols:
    print(f"\n⚠️ Warning: Missing columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    feature_columns = [col for col in feature_columns if col in df.columns]

# Handle missing values
X = df[feature_columns].fillna(0)
y = df['emission_level']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML model
model = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=15)
model.fit(X_train, y_train)

# Predict test set
pred = model.predict(X_test)

# Print evaluation
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print("\nAccuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# Feature importance
print("\nTop 5 Important Features:")
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head())

# Save model
model_dir = os.path.join(os.path.dirname(__file__), "../model")
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(feature_columns, os.path.join(model_dir, "features.pkl"))

print("\n✓ Model saved successfully!")
print("✓ Features saved successfully!")
