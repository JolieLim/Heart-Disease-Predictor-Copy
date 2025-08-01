import streamlit as st
import pandas as pd
import joblib

# --- Load model, scaler, and expected columns ---
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("model_columns.pkl") 

# --- BMI Calculation ---
def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return round(weight_kg / (height_m ** 2), 1)

# --- UI Form ---
def show_user_form():
    st.title("💓 Heart Disease Predictor")
    st.markdown("### Please enter your details")

    gender = st.selectbox("Sex", ["Female", "Male"])
    age_category = st.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54',
        '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
    ])
    race = st.selectbox("Race", [
        'White', 'Black', 'Asian', 'American Indian/Alaskan Native',
        'Hispanic', 'Other'
    ])
    smoking = st.selectbox("Smoked 100+ cigarettes?", ['No', 'Yes'])
    alcohol = st.selectbox("Heavy alcohol consumption?", ['No', 'Yes'])
    stroke = st.selectbox("Ever had a stroke?", ['No', 'Yes'])
    physical_activity = st.selectbox("Physical activity in past 30 days?", ['Yes', 'No'])
    diff_walking = st.selectbox("Difficulty walking/climbing stairs?", ['No', 'Yes'])
    diabetic = st.selectbox("Are you diabetic?", ['No', 'Yes', 'No, borderline diabetes', 'Yes (during pregnancy)'])
    physical_health = st.slider("Days of poor physical health (0-30)", 0, 30, 0)
    mental_health = st.slider("Days of poor mental health (0-30)", 0, 30, 0)
    sleep_time = st.slider("Average hours of sleep per day", 0, 24, 7)
    general_health = st.selectbox("General health rating", ['Excellent', 'Very good', 'Good', 'Fair', 'Poor'])
    asthma = st.selectbox("Do you have asthma?", ['No', 'Yes'])
    kidney_disease = st.selectbox("Do you have kidney disease?", ['No', 'Yes'])
    skin_cancer = st.selectbox("Do you have skin cancer?", ['No', 'Yes'])
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
    weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)

    bmi = calculate_bmi(weight_kg, height_cm)
    st.markdown(f"**Calculated BMI:** `{bmi}`")

    return {
        "Sex": gender,
        "AgeCategory": age_category,
        "Race": race,
        "Smoking": smoking,
        "AlcoholDrinking": alcohol,
        "Stroke": stroke,
        "PhysicalActivity": physical_activity,
        "DiffWalking": diff_walking,
        "Diabetic": diabetic,
        "PhysicalHealth": physical_health,
        "MentalHealth": mental_health,
        "SleepTime": sleep_time,
        "GenHealth": general_health,
        "Asthma": asthma,
        "KidneyDisease": kidney_disease,
        "SkinCancer": skin_cancer,
        "BMI": bmi
    }

# --- Prediction Function ---
def predict_heart_disease(user_input):
    df = pd.DataFrame([user_input])

    # Binary mappings
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    for col in ['Sex', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalActivity',
                'DiffWalking', 'Asthma', 'KidneyDisease', 'SkinCancer']:
        df[col] = df[col].map(binary_map)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align columns to match training set
    df = df.reindex(columns=expected_columns, fill_value=0)

    # Scale features
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return prediction, probability

# --- Main ---
form_data = show_user_form()

if st.button("💡 Predict"):
    prediction, probability = predict_heart_disease(form_data)
    if prediction == 1:
        st.error(f"⚠️ The model predicts a **HIGH RISK** of heart disease. (Probability: {probability:.2%})")
    else:
        st.success(f"✅ The model predicts **LOW RISK** of heart disease. (Probability: {probability:.2%})")
