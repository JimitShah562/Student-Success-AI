import streamlit as st
import joblib
import numpy as np
import pandas as pd 

model = joblib.load("Models/student_model.pkl")

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓")
st.title("Student Success Predictor")
st.write("Fill in the student data to predict your result !")

# -- Input -- 
study_hours = st.slider("📘 Study Hours", 0.0, 10.0, 5.0)
attendance = st.slider("📅 Attendance (%)", 0, 100, 70)
assignments_done = st.selectbox("📑 Assignments Completed?", ("Yes", "No"))
internet_access = st.selectbox("🌐 Internet Access?", ("Yes", "No"))
parent_support = st.selectbox("👨‍👩‍👧 Parental Support?", ("Yes", "No"))
health_issues = st.selectbox("💊 Health Issues?", ("Yes", "No"))


# -- Convert Input --
assignments_done = 1 if assignments_done == "Yes" else 0
internet_access = 1 if internet_access == "Yes" else 0
parent_support = 1 if parent_support == "Yes" else 0
health_issues = 1 if health_issues == "Yes" else 0


# -- Prediction -- 
if st.button("Predict Your Result"):
    input_data = pd.DataFrame([
         
         [  study_hours, attendance, assignments_done,
               internet_access, parent_support, health_issues
         ]      
                                  
    ], columns = ['study_hours', 'attendance', 'assignments_done', 'internet_access', 'parent_support', 'health_issues'])


    result = model.predict(input_data)[0]
    if result == 1:
         st.success("🎉 The student is likely to **Pass** ✅")
    else :
         st.error("⚠️ The student is likely to **Fail** ❌")