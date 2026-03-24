# ============================================
# 1. IMPORT LIBRARIES
# ============================================

import streamlit as st
import numpy as np
import joblib


# ============================================
# 2. LOAD MODEL + COLUMNS
# ============================================

model = joblib.load("heart_model.pkl")
columns = joblib.load("columns.pkl")


# ============================================
# 3. PAGE CONFIG
# ============================================

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk")

st.markdown("---")


# ============================================
# 4. INPUT SECTION
# ============================================

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)

    sex = st.selectbox("Sex", ["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    cp_dict = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp_label = st.selectbox("Chest Pain Type", list(cp_dict.keys()))
    chest_pain_type = cp_dict[cp_label]

    resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)

    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 400, 200)
    st.caption("Normal cholesterol is below 200 mg/dL")

    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    fasting_blood_sugar = 1 if fasting_blood_sugar == "Yes" else 0


with col2:
    ecg_dict = {
        "Normal": 0,
        "ST-T Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    ecg_label = st.selectbox("Resting ECG", list(ecg_dict.keys()))
    resting_ecg = ecg_dict[ecg_label]

    max_heart_rate = st.number_input("Max Heart Rate", 60, 220, 150)

    exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exercise_induced_angina = 1 if exercise_induced_angina == "Yes" else 0

    st_depression = st.number_input("ST Depression", 0.0, 6.0, 1.0)

    slope_dict = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope_label = st.selectbox("ST Slope", list(slope_dict.keys()))
    st_slope = slope_dict[slope_label]

    num_major_vessels = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])

    thal_dict = {
        "Normal": 0,
        "Fixed Defect": 1,
        "Reversible Defect": 2,
        "Unknown": 3
    }
    thal_label = st.selectbox("Thalassemia", list(thal_dict.keys()))
    thalassemia = thal_dict[thal_label]


# ============================================
# 5. PREDICTION
# ============================================

if st.button("🔍 Predict"):

    # Input validation
    if cholesterol < 120 or cholesterol > 400:
        st.warning("⚠️ Please check cholesterol value")

    input_data = np.array([[ 
        age,
        sex,
        chest_pain_type,
        resting_blood_pressure,
        cholesterol,
        fasting_blood_sugar,
        resting_ecg,
        max_heart_rate,
        exercise_induced_angina,
        st_depression,
        st_slope,
        num_major_vessels,
        thalassemia
    ]])

    # Match column order
    input_dict = dict(zip(columns, input_data[0]))
    input_array = np.array([list(input_dict.values())])

    # Prediction
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)

    st.markdown("---")
    st.subheader("🩺 Prediction Result")

    # ============================================
    # RESULT DISPLAY
    # ============================================

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({probability[0][1]*100:.2f}%)")
        st.warning("Consult a doctor for further evaluation")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({probability[0][0]*100:.2f}%)")
        st.info("Maintain a healthy lifestyle and regular checkups")

    # ============================================
    # PROBABILITY
    # ============================================

    st.markdown("### 📊 Prediction Probability")
    st.write(f"Disease: {probability[0][1]*100:.2f}%")
    st.write(f"No Disease: {probability[0][0]*100:.2f}%")

    # ============================================
    # CONFIDENCE LOGIC
    # ============================================

    st.markdown("### 📈 Confidence Level")
    confidence = max(probability[0])
    st.progress(float(confidence))
    st.write(f"Model Confidence: {confidence*100:.2f}%")

    if confidence >= 0.8:
        st.success("🟢 High confidence prediction")
    elif confidence >= 0.6:
        st.warning("🟡 Moderate confidence - monitor condition")
    else:
        st.error("🔴 Low confidence - prediction uncertain")

    # Borderline case detection
    if 0.4 < probability[0][1] < 0.6:
        st.warning("⚠️ Borderline case – prediction is not very strong")

    # ============================================
    # EXPLANATION (IMPROVED)
    # ============================================

    st.markdown("### 🧠 Explanation")

    risk_factors = []

    if age >= 50:
        risk_factors.append("Higher age")

    if cholesterol >= 200:
        risk_factors.append("Borderline or high cholesterol")

    if resting_blood_pressure >= 130:
        risk_factors.append("Elevated blood pressure")

    if max_heart_rate < 100:
        risk_factors.append("Low heart rate")

    if exercise_induced_angina == 1:
        risk_factors.append("Exercise-induced angina")

    if st_depression >= 1:
        risk_factors.append("Abnormal ST depression")

    if len(risk_factors) > 0:
        st.write("⚠️ Key contributing factors based on your inputs:")
        for factor in risk_factors:
            st.write(f"- {factor}")
    else:
        st.success("✅ No significant risk factors detected")

    # ============================================
    # DISCLAIMER
    # ============================================

    st.markdown("---")
    st.info("⚠️ This prediction is based on machine learning and should not replace professional medical advice.")