"""
House Price Prediction - Streamlit App
----------------------------------------
Inputs: Area, BHK, Bathroom, Parking, Per_Sqft
Run with: streamlit run house_price_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Prediction")
st.write(
    "Enter the property details below to get an estimated house price."
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "house_price_model2.pkl"

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model(MODEL_PATH)

if model is None:
    st.warning(
        f"⚠️ No trained model found at '{MODEL_PATH}'. "
        "Using a simple fallback formula for demonstration purposes. "
        "Place your trained model (scikit-learn, saved with joblib) "
        "in the same folder as this script, named 'house_price_model.pkl', "
        "to get real predictions."
    )

# -----------------------------
# Sidebar - About
# -----------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This app predicts house prices based on:\n"
        "- Area (sq ft)\n"
        "- BHK (bedrooms)\n"
        "- Bathrooms\n"
        "- Parking spaces\n"
        "- Price per sq ft"
    )
    st.markdown("---")
    st.caption("Built with Streamlit")

# -----------------------------
# Input Form
# -----------------------------
st.subheader("Property Details")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area (sq ft)",
        min_value=100.0,
        max_value=20000.0,
        value=1000.0,
        step=50.0
    )
    bhk = st.number_input(
        "BHK (Bedrooms)",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )
    bathroom = st.number_input(
        "Bathroom",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col2:
    parking = st.number_input(
        "Parking",
        min_value=0,
        max_value=10,
        value=1,
        step=1
    )
    per_sqft = st.number_input(
        "Per Sqft Price (₹)",
        min_value=100.0,
        max_value=100000.0,
        value=5000.0,
        step=100.0
    )

st.markdown("---")

# -----------------------------
# Prediction
# -----------------------------
input_df = pd.DataFrame({
    "Area": [area],
    "BHK": [bhk],
    "Bathroom": [bathroom],
    "Parking": [parking],
    "Per_Sqft": [per_sqft]
})

st.subheader("Input Summary")
st.dataframe(input_df, use_container_width=True)

def fallback_prediction(area, bhk, bathroom, parking, per_sqft):
    """Simple heuristic used only when no trained model is available."""
    base_price = area * per_sqft
    adjustment = (bhk * 50000) + (bathroom * 20000) + (parking * 15000)
    return base_price + adjustment

if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        if model is not None:
            prediction = model.predict(input_df)[0]
        else:
            prediction = fallback_prediction(area, bhk, bathroom, parking, per_sqft)

        st.success(f"### 💰 Estimated Price: ₹ {prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption(
    "Note: Predictions are estimates. Accuracy depends on the quality "
    "of the trained model used."
)