import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model, scaler, and column list
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
column_order = joblib.load("column_order.pkl")

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("❤️ Heart Disease Predictor")
st.markdown("Answer the following to predict your risk level:")

# Numeric inputs
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
phys_health = st.slider("Days of poor physical health (last 30 days)", 0, 30)
ment_health = st.slider("Days of poor mental health (last 30 days)", 0, 30)
sleep_time = st.slider("Average Sleep Time (hrs/day)", 0, 24)

# Binary/Categorical inputs (used for one-hot encoding)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
alcohol = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
stroke = st.selectbox("Have you ever had a stroke?", ["No", "Yes"])
diff_walking = st.selectbox("Do you have difficulty walking?", ["No", "Yes"])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.selectbox("Age Category", [
    "18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
    "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"
])
race = st.selectbox("Race", ["White", "Black", "Asian", "American Indian/Alaskan Native", "Other", "Hispanic"])
diabetic = st.selectbox("Are you diabetic?", ["No", "Yes"])
phys_act = st.selectbox("Do you engage in physical activity?", ["No", "Yes"])
gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
asthma = st.selectbox("Do you have asthma?", ["No", "Yes"])
kidney_disease = st.selectbox("Kidney disease?", ["No", "Yes"])
skin_cancer = st.selectbox("Skin cancer?", ["No", "Yes"])

# -------------------------------
# Build Input Row with Dummy Values
# -------------------------------

# Initial input with numeric fields
input_dict = {
    "BMI": bmi,
    "PhysicalHealth": phys_health,
    "MentalHealth": ment_health,
    "SleepTime": sleep_time,
    f"Smoking_{smoking}": 1,
    f"AlcoholDrinking_{alcohol}": 1,
    f"Stroke_{stroke}": 1,
    f"DiffWalking_{diff_walking}": 1,
    f"Sex_{sex}": 1,
    f"AgeCategory_{age}": 1,
    f"Race_{race}": 1,
    f"Diabetic_{diabetic}": 1,
    f"PhysicalActivity_{phys_act}": 1,
    f"GenHealth_{gen_health}": 1,
    f"Asthma_{asthma}": 1,
    f"KidneyDisease_{kidney_disease}": 1,
    f"SkinCancer_{skin_cancer}": 1
}

# Fill in all required columns with 0 if missing
input_df = pd.DataFrame([input_dict])
for col in column_order:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder to match training data
input_df = input_df[column_order]

# -------------------------------
# Predict
# -------------------------------
if st.button("🔮 Predict"):
    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction == 1:
        st.error(f"High Risk of Heart Disease (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk of Heart Disease (Probability: {prob:.2f})")
