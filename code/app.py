import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Load model and features
model_path = os.path.join(os.path.dirname(__file__), "../model/model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "../model/features.pkl")

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

st.title("🌍 Tourism Carbon Footprint Calculator")
st.write("This app predicts the overall carbon emission level for a tourist's trip based on comprehensive travel data.")

# Create organized input sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚗 Trip Details")
    trip_days = st.number_input("Trip Duration (days)", 1, 30, 6)
    distance_km = st.number_input("Travel Distance (km)", 1.0, 1000.0, 54.8)
    occupancy = st.number_input("Vehicle Occupancy", 1, 100, 27)
    
    st.subheader("🌍 Environmental Factors")
    congestion_factor = st.slider("Congestion Factor (0.8-1.2)", 0.8, 1.2, 0.91, 0.01)
    terrain_factor = st.slider("Terrain Factor (1.0-1.3)", 1.0, 1.3, 1.15, 0.01)

with col2:
    st.subheader("🏨 Accommodation")
    nightly_kwh_est = st.number_input("Nightly Electricity (kWh)", 1.0, 200.0, 8.44)
    grid_ef_kgCO2_per_kWh = st.number_input("Grid Carbon Efficiency (kgCO2/kWh)", 0.01, 1.0, 0.07, 0.01)
    
    st.subheader("🍽️ Food & Waste")
    food_emissions_kgCO2 = st.number_input("Food Emissions (kgCO2)", 0.0, 100.0, 16.884)
    waste_emissions_kgCO2 = st.number_input("Waste Emissions (kgCO2)", 0.0, 50.0, 2.998)
    plastic_emissions_kgCO2 = st.number_input("Plastic Emissions (kgCO2)", 0.0, 10.0, 0.3)

col3, col4 = st.columns(2)

with col3:
    st.subheader("🚌 Transport Emissions")
    transport_emissions_kgCO2 = st.number_input("Transport Emissions (kgCO2)", 0.0, 500.0, 0.037)

with col4:
    st.subheader("⚡ Accommodation Emissions")
    accommodation_elec_kgCO2 = st.number_input("Electricity Emissions (kgCO2)", 0.0, 200.0, 4.805)
    accommodation_gen_kgCO2 = st.number_input("Generator Emissions (kgCO2)", 0.0, 200.0, 11.095)

# Recommendation Function
def get_recommendation(level):
    if level == "low":
        return (
            "✅ Low emissions. Keep using eco-friendly transport, reusable items, "
            "and sustainable practices."
        )
    elif level == "medium":
        return (
            "⚠️ Medium emissions. Reduce meat consumption, lower electricity use, "
            "minimize plastic usage, and opt for public transport."
        )
    elif level == "high":
        return (
            "🔴 High emissions! Switch to EV/public transport, reduce accommodation energy use, "
            "eat more vegetarian meals, avoid plastic, and choose eco-certified hotels."
        )
    else:
        return "No recommendation available."

def align_features(input_df, expected_columns):
    df_local = pd.get_dummies(input_df)
    for col in expected_columns:
        if col not in df_local.columns:
            df_local[col] = 0
    df_local = df_local[expected_columns]
    return df_local

# Prediction Button
if st.button("🔮 Predict Emission Level", use_container_width=True):
    # Prepare input data in correct order
    input_dict = {
        'trip_days': trip_days,
        'distance_km': distance_km,
        'occupancy': occupancy,
        'congestion_factor': congestion_factor,
        'terrain_factor': terrain_factor,
        'nightly_kwh_est': nightly_kwh_est,
        'grid_ef_kgCO2_per_kWh': grid_ef_kgCO2_per_kWh,
        'food_emissions_kgCO2': food_emissions_kgCO2,
        'waste_emissions_kgCO2': waste_emissions_kgCO2,
        'plastic_emissions_kgCO2': plastic_emissions_kgCO2,
        'transport_emissions_kgCO2': transport_emissions_kgCO2,
        'accommodation_elec_kgCO2': accommodation_elec_kgCO2,
        'accommodation_gen_kgCO2': accommodation_gen_kgCO2
    }
    
    input_df = pd.DataFrame([input_dict])
    aligned_df = align_features(input_df, feature_columns)
    input_data = aligned_df.values
    
    prediction = model.predict(input_data)[0]

    st.subheader(f"🌡 Emission Level: **{prediction.upper()}**")
    st.write("### 📌 Recommendation:")
    st.success(get_recommendation(prediction))

