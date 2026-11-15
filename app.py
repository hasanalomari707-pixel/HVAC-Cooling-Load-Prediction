import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Professional Dashboard Configuration
st.set_page_config(
    page_title="HVAC Intelligence Dashboard",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dashboard Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0B1220;
        color: #F8FAFC;
    }

    h1, h2, h3 {
        color: #38BDF8 !important;
    }

    div[data-testid="stMetric"] {
        background-color: #172338;
        border: 1px solid #334155;
        padding: 18px;
        border-radius: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #CBD5E1;
    }

    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
    }

    section[data-testid="stSidebar"] {
        background-color: #111C30;
    }
</style>
""", unsafe_allow_html=True)
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
st.sidebar.subheader("🏢 Building Parameters")
col1 = st.sidebar.container()
col2 = st.sidebar.container()
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
if st.sidebar.button(
    "🔍 Predict Cooling Load",
    use_container_width=True
):
    prediction = model.predict(input_data)

    st.success(
        f"Predicted Cooling Load: {prediction[0]:.2f}"
    )

st.divider()

st.caption(
    "HVAC Machine Learning Project | Mechanical Engineering"
)


# Feature Importance Chart

st.divider()
st.subheader("📊 Feature Importance")

features = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Distribution"
]

importance = model.feature_importances_

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(features, importance)

ax.set_xlabel("Importance")
ax.set_title("Factors Affecting HVAC Cooling Load")

ax.invert_yaxis()

st.pyplot(fig)
plt.close(fig)

# Actual vs Predicted Cooling Load
st.divider()

st.subheader("📈 Actual vs Predicted Cooling Load")

comparison = pd.read_csv("comparison.csv")

st.line_chart(
    comparison[["Actual", "Predicted"]],
    x_label="Test Sample",
    y_label="Cooling Load"
)

# Prediction Error Analysis
st.divider()

st.subheader("📉 Prediction Error Analysis")

comparison["Error"] = (
    comparison["Actual"] - comparison["Predicted"]
)

st.bar_chart(
    comparison["Error"],
    x_label="Test Sample",
    y_label="Prediction Error"
)

st.write(
    "This chart shows the difference between "
    "actual and predicted cooling loads."
)

# Machine Learning Model Comparison
st.divider()

st.subheader("🤖 Machine Learning Model Comparison")

model_results = pd.read_csv("model_comparison.csv")

st.dataframe(model_results, use_container_width=True)

st.subheader("Mean Absolute Error Comparison")

st.bar_chart(
    model_results.set_index("Model")["MAE"],
    y_label="MAE (Lower is Better)"
)

st.subheader("R² Score Comparison")

st.bar_chart(
    model_results.set_index("Model")["R2"],
    y_label="R² (Higher is Better)"
)