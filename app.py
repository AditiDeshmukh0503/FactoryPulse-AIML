
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="FactoryPulse AI",
    page_icon="🏭",
    layout="wide"
)

# Load trained model
artifact = joblib.load("factorypulse_model.pkl")

model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]

# Header
st.title("🏭 FactoryPulse AI")
st.subheader("AI-Powered Predictive Maintenance System")
st.caption("Intelligent machine failure risk assessment")

st.divider()

# Sidebar
st.sidebar.header("Machine Parameters")

product_type = st.sidebar.selectbox(
    "Product Type",
    ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)",
    min_value=290.0,
    max_value=310.0,
    value=300.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)",
    min_value=300.0,
    max_value=320.0,
    value=310.0
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)",
    min_value=1000,
    max_value=3000,
    value=1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=100.0,
    value=40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (minutes)",
    min_value=0,
    max_value=300,
    value=100
)

# Prepare input in the exact training feature order
input_data = {
    "Air temperature [K]": air_temp,
    "Process temperature [K]": process_temp,
    "Rotational speed [rpm]": rotational_speed,
    "Torque [Nm]": torque,
    "Tool wear [min]": tool_wear,
    "Type_L": int(product_type == "L"),
    "Type_M": int(product_type == "M")
}

input_df = pd.DataFrame(
    [[input_data.get(feature, 0) for feature in features]],
    columns=features
)

st.header("Machine Health Assessment")

if st.button("🔍 Analyse Machine", use_container_width=True):

    probability = model.predict_proba(input_df)[0][1]
    prediction = int(probability >= threshold)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Failure Score",
            f"{probability * 100:.2f}%"
        )

    with col2:
        st.metric(
            "Decision Threshold",
            f"{threshold * 100:.0f}%"
        )

    st.progress(float(probability))

    if prediction == 1:
        st.error("⚠️ POTENTIAL FAILURE DETECTED")

        st.warning(
            "The model score has crossed the configured "
            "decision threshold. Arrange a machine inspection."
        )

        st.write("Recommended actions:")
        st.write("- Inspect machine operating conditions.")
        st.write("- Check torque and rotational speed.")
        st.write("- Inspect tool wear.")
        st.write("- Schedule preventive maintenance.")

    else:
        st.success("✅ NO FAILURE FLAGGED BY THE MODEL")

        st.info(
            "The model score is below the configured threshold. "
            "Continue routine monitoring and scheduled maintenance."
        )

    st.caption(
        "This is a machine-learning risk score, not a calibrated "
        "probability or a guarantee of machine safety."
    )

st.divider()

# Model information
st.header("Model Information")

c1, c2, c3 = st.columns(3)

c1.metric("Algorithm", "Random Forest")
c2.metric("Test Accuracy", "97.70%")
c3.metric("Test F1-score", "63.49%")

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    importance["Feature"],
    importance["Importance"]
)

ax.set_xlabel("Importance")
ax.set_title("Machine Failure Prediction - Feature Importance")

st.pyplot(fig)

st.divider()

st.caption(
    "FactoryPulse AI | Academic Predictive Maintenance Prototype"
)
