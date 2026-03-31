import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model
model = joblib.load("heart_disease_pipeline.joblib")

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# 🔥 FULL UI CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #134e5e, #71b280);
    color: white;
}

/* Header */
.header {
    text-align: center;
    padding: 20px;
    font-size: 32px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg, #1e3a5f, #2b5876);
    border-radius: 15px;
    margin-bottom: 20px;
}

/* Labels */
label {
    color: black !important;
    font-weight: bold !important;
}

/* Inputs */
.stNumberInput>div>div>input {
    background-color: #1e222b;
    color: white;
    border-radius: 8px;
    border: 1px solid #555;
}

/* Predict Button */
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 200px;
    font-weight: bold;
}

/* Download Button */
.stDownloadButton>button {
    background-color: #0b3d3d;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: 1px solid #00bcd4;
}
.stDownloadButton>button:hover {
    background-color: #145f5f;
}

/* Cards */
.card {
    background: linear-gradient(135deg, #1e3a5f, #2b5876);
    padding: 20px;
    border-radius: 15px;
    margin-top: 15px;
    box-shadow: 0 0 15px rgba(0, 150, 255, 0.2);
    border: 1px solid rgba(0, 150, 255, 0.3);
}

/* Card text */
.card h3, .card p, .card li {
    color: #e6f2ff !important;
}

/* Hover */
.card:hover {
    transform: scale(1.01);
    transition: 0.2s;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header">❤️ Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)

# Inputs
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    trestbps = st.number_input("Resting Blood Pressure", 50, 250, 120)
    restecg = st.number_input("Rest ECG (0-2)", 0, 2, 0)
    oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, step=0.1)
    thal = st.number_input("Thal (0-2)", 0, 2, 1)

with col2:
    sex = st.number_input("Sex (1=male,0=female)", 0, 1, 1)
    chol = st.number_input("Cholesterol", 100, 600, 200)
    thalach = st.number_input("Max Heart Rate", 50, 220, 150)
    slope = st.number_input("Slope (0-2)", 0, 2, 1)

with col3:
    cp = st.number_input("Chest Pain (0-3)", 0, 3, 0)
    fbs = st.number_input("Fasting Blood Sugar (1/0)", 0, 1, 0)
    exang = st.number_input("Exercise Angina (1/0)", 0, 1, 0)
    ca = st.number_input("Major Vessels (0-3)", 0, 3, 0)

# History
if "history" not in st.session_state:
    st.session_state.history = []

# Predict
if st.button("Predict"):
    try:
        input_data = np.array([[ 
            age, sex, cp, trestbps, chol,
            fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal
        ]])

        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)
        risk = proba[0][1] * 100

        # 🔥 RESULT BOX (VISIBLE)
        if prediction[0] == 1:
            result = "⚠️ The person has Heart Disease"
            st.markdown(f"""
            <div style="background-color:#ffcccb;padding:12px;border-radius:10px;color:black;font-weight:bold">
            {result}
            </div>
            """, unsafe_allow_html=True)
        else:
            result = "✅ The person does NOT have Heart Disease"
            st.markdown(f"""
            <div style="background-color:#b2dfdb;padding:12px;border-radius:10px;color:black;font-weight:bold">
            {result}
            </div>
            """, unsafe_allow_html=True)

        # Risk
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Risk Analysis")
        st.write(f"Risk Percentage: {risk:.2f}%")

        if risk < 30:
            st.markdown("<p style='color:black;font-weight:bold'>🟢 Low Risk</p>", unsafe_allow_html=True)
        elif risk < 60:
            st.markdown("<p style='color:black;font-weight:bold'>🟡 Medium Risk</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:black;font-weight:bold'>🔴 High Risk</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Tips
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if prediction[0] == 1:
            st.subheader("💊 Care Tips")
            st.write("""
            - Maintain healthy diet  
            - Exercise regularly  
            - Avoid smoking & alcohol  
            - Monitor BP  
            - Consult doctor  
            """)
        else:
            st.subheader("🛡️ Prevention Tips")
            st.write("""
            - Balanced diet  
            - Stay active  
            - Avoid stress  
            - Regular checkups  
            """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Alerts
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🧠 Health Alerts")

        alert = False
        if chol > 240:
            st.markdown("<p style='color:black'>⚠️ High Cholesterol</p>", unsafe_allow_html=True)
            alert = True
        if trestbps > 140:
            st.markdown("<p style='color:black'>⚠️ High Blood Pressure</p>", unsafe_allow_html=True)
            alert = True
        if oldpeak > 2:
            st.markdown("<p style='color:black'>⚠️ ST Depression</p>", unsafe_allow_html=True)
            alert = True
        if thalach < 100:
            st.markdown("<p style='color:black'>⚠️ Low Heart Rate</p>", unsafe_allow_html=True)
            alert = True

        if not alert:
            st.markdown("<p style='color:black;font-weight:bold'>✅ No critical alerts</p>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Summary
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Patient Summary")
        st.write(f"""
        - Age: {age}
        - Sex: {'Male' if sex == 1 else 'Female'}
        - Cholesterol: {chol}
        - Blood Pressure: {trestbps}
        - Max Heart Rate: {thalach}
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Score
        score = 100
        if chol > 240: score -= 20
        if trestbps > 140: score -= 20
        if oldpeak > 2: score -= 20
        if exang == 1: score -= 20

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("❤️ Health Score")
        st.write(f"{score}/100")
        st.markdown('</div>', unsafe_allow_html=True)

        # Habits
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📅 Daily Habits")
        st.write("""
        - 🚶 Walk daily  
        - 🥗 Eat healthy  
        - 💧 Drink water  
        - 😴 Sleep well  
        - 🧘 Manage stress  
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # Feature Importance
        feature_names = ['age','sex','cp','trestbps','chol','fbs','restecg',
                         'thalach','exang','oldpeak','slope','ca','thal']

        importances = model.named_steps['model'].feature_importances_

        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Feature Importance")
        st.bar_chart(df_imp.set_index('Feature'))
        st.markdown('</div>', unsafe_allow_html=True)

        # Report
        report = f"""
Heart Disease Report

Result: {result}
Risk: {risk:.2f}%
Age: {age}
Cholesterol: {chol}
BP: {trestbps}
"""
        st.download_button("📥 Download Report", report, file_name="report.txt")

        # History
        st.session_state.history.append({
            "Age": age,
            "Result": result,
            "Risk": f"{risk:.2f}%"
        })

    except Exception as e:
        st.error(f"Error: {str(e)}")

# History
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📜 Prediction History")
st.write(st.session_state.history)
st.markdown('</div>', unsafe_allow_html=True)
