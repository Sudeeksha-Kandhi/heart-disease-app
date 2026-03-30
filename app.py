import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("random_forest_heart_disease_model.pkl") # rename to your actual file name

st.title("Heart Disease Prediction (Random Forest)")

st.write("Enter patient details:")

# Inputs (based on Cleveland dataset)
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = True, 0 = False)", [0,1])
restecg = st.selectbox("Rest ECG (0-2)", [0,1,2])
thalach = st.number_input("Max Heart Rate")
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0,1])
oldpeak = st.number_input("ST Depression", step=0.1)
slope = st.selectbox("Slope (0-2)", [0,1,2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal = st.selectbox("Thal (1,2,3)", [1,2,3])

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High chance of Heart Disease")
    else:
        st.success("✅ Low chance of Heart Disease")