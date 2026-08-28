#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Breast Cancer Prediction", page_icon="🩺")

st.title("🩺 Breast Cancer Diagnosis Prediction")

st.write("Enter the feature values below to predict the diagnosis.")

# User Inputs
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, format="%.4f")
area_worst = st.number_input("Area Worst", min_value=0.0, format="%.4f")
radius_worst = st.number_input("Radius Worst", min_value=0.0, format="%.4f")
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, format="%.4f")
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, format="%.4f")
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, format="%.4f")
area_mean = st.number_input("Area Mean", min_value=0.0, format="%.4f")
radius_mean = st.number_input("Radius Mean", min_value=0.0, format="%.4f")
area_se = st.number_input("Area SE", min_value=0.0, format="%.4f")
concavity_mean = st.number_input("Concavity Mean", min_value=0.0, format="%.4f")

if st.button("Predict Diagnosis"):

    input_data = np.array([[
        perimeter_worst,
        area_worst,
        radius_worst,
        concave_points_worst,
        concave_points_mean,
        perimeter_mean,
        area_mean,
        radius_mean,
        area_se,
        concavity_mean
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Display Result
    if prediction[0] == "M" or prediction[0] == 1:
        st.error("🔴 Diagnosis: Malignant")
    else:
        st.success("🟢 Diagnosis: Benign")

