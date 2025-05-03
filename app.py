import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the saved model and encoders
model = joblib.load("sleep_disorder_model.pkl")
le_target = joblib.load("label_encoder.pkl")

# If you saved encoders for input features, you can load them here
# input_encoders = joblib.load("input_label_encoders.pkl")

st.title("Sleep Disorder Prediction App 💤")

# User inputs
gender = st.selectbox("Gender", ['Male', 'Female'])
age = st.number_input("Age", min_value=0, max_value=100)
occupation = st.selectbox("Occupation", ['Doctor', 'Engineer', 'Nurse', 'Teacher', 'Lawyer', 'Accountant', 'Salesperson', 'Scientist', 'Artist', 'Software Engineer'])  # customize this based on training data

# Add more inputs below to match the **exact training features**
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, max_value=24.0)
quality_of_sleep = st.slider("Quality of Sleep (1-10)", 1, 10)
physical_activity = st.slider("Physical Activity (hours/day)", 0, 5)
stress_level = st.slider("Stress Level (1-10)", 1, 10)
bmi_category = st.selectbox("BMI Category", ['Normal', 'Overweight', 'Obese', 'Underweight'])
heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200)
daily_steps = st.number_input("Daily Steps", min_value=0, max_value=50000)

# Blood pressure fields if applicable
systolic = st.number_input("Systolic BP", min_value=50, max_value=250)
diastolic = st.number_input("Diastolic BP", min_value=30, max_value=150)

# Encode categorical inputs manually if you didn't save encoders
gender_val = 1 if gender == 'Male' else 0
occupation_val = hash(occupation) % 100  # Replace with real encoder or mapping
bmi_val = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3}[bmi_category]

# Arrange the features in the correct order as per your model training
features = np.array([[
    gender_val,
    age,
    occupation_val,
    sleep_duration,
    quality_of_sleep,
    physical_activity,
    stress_level,
    bmi_val,
    heart_rate,
    daily_steps,
    systolic,
    diastolic
]])

# Keep only the features used during training (drop extra fields if needed)
features = features[:, :model.n_features_in_]

if st.button("Predict Sleep Disorder"):
    prediction = model.predict(features)
    prediction_label = le_target.inverse_transform(prediction)[0]
    st.subheader(f"Prediction: {prediction_label}")
\
    