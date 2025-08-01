import streamlit as st
import numpy as np
import pandas as pd
import joblib

# --- Load Model and Scaler ---
model = joblib.load("best_random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- BMI Calculation ---
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

# --- Input Form ---
def get_user_input():
    st.title("💓 Heart Disease Predictor")
    st.markdown("Please fill in the details below:")

    gender = st.selectbox("Sex", ["Female", "Male"])
    age = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])
    race = st.selectbox("Race", [
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native',
        'Hispanic', 'Other'
    ])
    general_health = st.selectbox("General Health", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    smoking = st.selectbox("Have you smoked 100+ cigarettes?", ['No', 'Yes'])
    alcohol = st.selectbox("Do you drink heavily?", ['No', 'Yes'])
    stroke = st.selectbox("Have you had a stroke?", ['No', 'Yes'])
    physical_activity = st.selectbox("Physical Activity (past 30 days)?", ['Yes', 'No'])
    diff_walking = st.selectbox("Difficulty walking?", ['No', 'Yes'])
    diabetic = st.selectbox("Are you diabetic?", ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    asthma = st.selectbox("Do you have asthma?", ['No', 'Yes'])
    kidney_disease = st.selectbox("Do you have kidney disease?", ['No', 'Yes'])
    skin_cancer = st.selectbox("Do you have skin cancer?", ['No', 'Yes'])
    physical_health = st.slider("Days of poor physical health (0-30)", 0, 30, 0)
    mental_health = st.slider("Days of poor mental health (0-30)", 0, 30, 0)
    sleep_time = st.slider("Average hours of sleep", 0, 24, 7)
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

    bmi = calculate_bmi(weight_kg, height_cm)
    st.markdown(f"**Your BMI:** `{bmi}`")

    return {
        "Sex": gender,
        "AgeCategory": age,
        "Race": race,
        "GenHealth": general_health,
        "Smoking": smoking,
        "AlcoholDrinking": alcohol,
        "Stroke": stroke,
        "PhysicalActivity": physical_activity,
        "DiffWalking": diff_walking,
        "Diabetic": diabetic,
        "Asthma": asthma,
        "KidneyDisease": kidney_disease,
        "SkinCancer": skin_cancer,
        "PhysicalHealth": physical_health,
        "MentalHealth": mental_health,
        "SleepTime": sleep_time,
        "BMI": bmi
    }

# --- Prediction Pipeline ---
def preprocess_input(form_data):
    # Convert to DataFrame
    df_input = pd.DataFrame([form_data])

    # Binary label encoding
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity', 'DiffWalking',
                'Asthma', 'KidneyDisease', 'SkinCancer', 'Sex']:
        df_input[col] = df_input[col].map(binary_map)

    # One-hot encoding for categorical variables
    cat_features = ['AgeCategory', 'Race', 'GenHealth', 'Diabetic']
    df_input = pd.get_dummies(df_input, columns=cat_features)

    # Align with training columns
    expected_cols = joblib.load("model_columns.pkl")  # this must be saved during training
    for col in expected_cols:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[expected_cols]

    return df_input

# --- Run App ---
form_data = get_user_input()

if st.button("Predict Heart Disease"):
    input_df = preprocess_input(form_data)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown("---")
    st.subheader("🩺 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ High Risk: The model predicts **Heart Disease**.")
    else:
        st.success("✅ Low Risk: The model predicts **No Heart Disease**.")
