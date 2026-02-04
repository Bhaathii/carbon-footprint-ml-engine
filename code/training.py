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

# REMOVE DATA LEAKAGE: Drop all emission component columns that directly contribute to total_trip_emissions_kgCO2
leak_cols = [
    'emission_level',
    'total_trip_emissions_kgCO2',
    'record_id',
    'trip_start_date',
    # Emission component columns (DATA LEAKAGE - these are used to calculate total emissions)
    'transport_emissions_kgCO2',
    'accommodation_elec_kgCO2',
    'accommodation_gen_kgCO2',
    'festival_gen_emissions_kgCO2',
    'pilgrimage_emissions_kgCO2',
    'food_emissions_kgCO2',
    'rice_emissions_kgCO2',
    'waste_emissions_kgCO2',
    'plastic_emissions_kgCO2'
]

print("\nRemoving data leakage columns:")
for col in leak_cols:
    if col in df.columns:
        print(f"  - {col}")

feature_df = df.drop(columns=[col for col in leak_cols if col in df.columns])

cat_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
feature_df = pd.get_dummies(feature_df, columns=cat_cols, drop_first=True)
feature_df = feature_df.fillna(0)

feature_columns = feature_df.columns.tolist()
X = feature_df
y = df['emission_level']

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ML model (optimized Random Forest without data leakage)
model = RandomForestClassifier(
    n_estimators=800,
    random_state=42,
    max_depth=20,
    max_features='sqrt',
    min_samples_leaf=2,
    min_samples_split=4,
    class_weight='balanced',
    n_jobs=-1
)
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

print("\nModel saved successfully!")
print("Features saved successfully!")
