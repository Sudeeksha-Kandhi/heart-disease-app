import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('random_forest_heart_disease_model.pkl', 'rb'))

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Dark theme + fix text visibility
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
label, .stTextInput label {
    color: white !important;
    font-weight: 500;
}
.stTextInput>div>div>input {
    background-color: #1e222b;
    color: white;
    border-radius: 8px;
    border: 1px solid #555;
}
.stSelectbox>div>div {
    background-color: #1e222b;
    color: white;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 200px;
    font-weight: bold;
}
h1 {
    text-align: center;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("Heart Disease Prediction")

col1, col2, col3 = st.columns(3)

# Using ranges/options based on UCI Heart Dataset
with col1:
    age = st.number_input("Age (29–77)", min_value=1, max_value=120, value=30)
    trestbps = st.number_input("Resting Blood Pressure (94–200 mm Hg)", min_value=50, max_value=250, value=120)
    restecg = st.selectbox("Resting ECG",
                           options=[0,1,2],
                           format_func=lambda x: {0:"Normal",1:"ST-T abnormality",2:"Left ventricular hypertrophy"}[x])
    oldpeak = st.number_input("ST Depression (0.0–6.2)", min_value=0.0, max_value=10.0, value=1.0)
    thal = st.selectbox("Thal",
                        options=[0,1,2],
                        format_func=lambda x: {0:"Normal",1:"Fixed defect",2:"Reversible defect"}[x])

with col2:
    sex = st.selectbox("Sex", options=[1,0], format_func=lambda x: "Male" if x==1 else "Female")
    chol = st.number_input("Cholesterol (126–564 mg/dl)", min_value=100, max_value=600, value=200)
    thalach = st.number_input("Max Heart Rate (71–202)", min_value=50, max_value=220, value=150)
    slope = st.selectbox("Slope",
                         options=[0,1,2],
                         format_func=lambda x: {0:"Upsloping",1:"Flat",2:"Downsloping"}[x])

with col3:
    cp = st.selectbox("Chest Pain Type",
                      options=[0,1,2,3],
                      format_func=lambda x: {0:"Typical angina",1:"Atypical angina",2:"Non-anginal pain",3:"Asymptomatic"}[x])
    fbs = st.selectbox("Fasting Blood Sugar > 120",
                       options=[0,1],
                       format_func=lambda x: "False" if x==0 else "True")
    exang = st.selectbox("Exercise Induced Angina",
                         options=[0,1],
                         format_func=lambda x: "No" if x==0 else "Yes")
    ca = st.selectbox("Major Vessels (0–3)", options=[0,1,2,3])

result = ""

if st.button("Predict"):
    try:
        input_data = np.array([[
            float(age), float(sex), float(cp), float(trestbps), float(chol),
            float(fbs), float(restecg), float(thalach), float(exang),
            float(oldpeak), float(slope), float(ca), float(thal)
        ]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "⚠️ The person has Heart Disease"
        else:
            result = "✅ The person does NOT have Heart Disease"

    except Exception as e:
        result = f"Error: {str(e)}"

st.success(result)
