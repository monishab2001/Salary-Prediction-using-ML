import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "" # Specify the path to the saved model here

# Loading the trained model
model = joblib.load(os.path.join(MODEL_PATH, "salary_prediction_model.pkl"))

# Loading feature columns & job roles
job_roles = joblib.load(os.path.join(MODEL_PATH,"job_titles.pkl"))
expected_columns = joblib.load(os.path.join(MODEL_PATH,"feature_columns.pkl"))

# Encoding mappings
education_mapping = {"High School": 1, "Bachelor's": 2, "Master's": 3, "PhD": 4}
gender_mapping = {"Female": 0, "Male": 1, "Others" : 2} #Label encoder assigns numbers in alphabetical order

# Streamlit UI
st.title("💰 Salary Prediction App")
st.write("Enter your details below to predict your estimated salary.")

# User Inputs
age = st.number_input("Enter Age:", min_value=18, max_value=65, step=1)
gender = st.selectbox("Select Gender:", list(gender_mapping.keys()))
education = st.selectbox("Select Education Level:", list(education_mapping.keys()))
experience = st.number_input("Enter Years of Experience:", min_value=0, max_value=50, step=1)
job = st.selectbox("Select Job Role:", job_roles)

# Converting user inputs to match model format
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender_mapping[gender]],
    "Education Level": [education_mapping[education]],
    "Years of Experience": [experience]
})

# Applying One-Hot Encoding to Job Role
for role in job_roles:
    input_data[f"{role}"] = [1 if job == role else 0]

# Ensuring correct column order
for col in expected_columns:
    if col not in input_data:
        input_data[col] = 0  # Adds missing one-hot encoded columns

input_data = input_data[expected_columns]

# Predicts Salary
if st.button("💡 Predict Salary"):
    salary_prediction = model.predict(input_data)
    st.success(f"📢 Estimated Salary: **${salary_prediction[0]:,.2f}**")

# Footer
st.markdown("---")
st.markdown("📌 *This app uses machine learning to estimate salaries based on user inputs.*")
