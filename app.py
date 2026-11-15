import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="HVAC ML Dashboard",
    page_icon="❄️",
    layout="wide"
)

# Load ML model
model = joblib.load("hvac_model.pkl")

# Dashboard header
st.title("❄️ HVAC Cooling Load Prediction")
st.write("Machine Learning for Mechanical Engineering")

st.divider()

# Model performance
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric("R² Score", "0.968")
col2.metric("MAE", "1.06")

st.divider()

# Building parameters
st.subheader("🏢 Building Parameters")

col1, col2 = st.columns(2)

with col1:
    x1 = st.slider("Relative Compactness", 0.62, 0.98, 0.75)
    x2 = st.slider("Surface Area", 514.5, 808.5, 673.75)
    x3 = st.slider("Wall Area", 245.0, 416.5, 318.5)
    x4 = st.slider("Roof Area", 110.25, 220.5, 147.0)

with col2:
    x5 = st.selectbox("Overall Height", [3.5, 7.0])
    x6 = st.selectbox("Orientation", [2, 3, 4, 5])
    x7 = st.selectbox("Glazing Area", [0.0, 0.1, 0.25, 0.4])
    x8 = st.selectbox("Glazing Distribution", [0, 1, 2, 3, 4, 5])

# Prepare model input
input_data = pd.DataFrame(
    [[x1, x2, x3, x4, x5, x6, x7, x8]],
    columns=["X1", "X2", "X3", "X4",
             "X5", "X6", "X7", "X8"]
)

# Prediction button
if st.button("🔍 Predict Cooling Load"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Cooling Load: {prediction[0]:.2f}"
    )

st.divider()

st.caption(
    "HVAC Machine Learning Project | Mechanical Engineering"
)
