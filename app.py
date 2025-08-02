import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model
with open('../src/modelos-random-forest.pkl', 'rb') as f:
    modelos = pickle.load(f)

model = modelos['rf_classifier_model']

class_dict = {
    0: "No Diabetes",
    1: "Diabetes"
}

# App layout
st.set_page_config(page_title="Diabetes Classifier", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter patient information below:")

# Input fields
pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=200.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=150.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, step=1.0)
insulin = st.number_input("Insulin (mu U/mL)", min_value=0.0, max_value=900.0, step=1.0)
bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age (years)", min_value=10, max_value=100, step=1)

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = int(model.predict(input_data)[0])
    label = class_dict[prediction]

    if prediction == 1:
        st.error("⚠️ Prediction: **Diabetes**")
    else:
        st.success("✅ Prediction: **No Diabetes**")
