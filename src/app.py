import os
import streamlit as st
import numpy as np
import joblib as jb
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Predict Z Factor", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "z_model_2025_04_15_v1.keras")
scaler_path = os.path.join(BASE_DIR, "model", "scaler_z_2025_04_15.pkl")

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = load_model(model_path)
    scaler = jb.load(scaler_path)
else:
    st.error(f"Model no found in the path: {model_path} or {scaler_path}")
    st.stop()

st.title("🧠 Predict Compressibility Z Factor")

st.markdown("""
            Model trained with TensorFlow to predict gas compressibility Z factor from pressure, temperature, and specific gravity.
            """)

pressure = st.number_input("Pressure (psia): ", min_value=1.0, step=100.0)
temperature = st.number_input("Temperature (oR): ", min_value=10.0, step=10.0)
sg = st.number_input("Specific Gravity: ", min_value=0.1, step=0.1, max_value=1.0)

if st.button("📈 Predict Z Factor"):
    input_array = np.array([[pressure, temperature, sg]])
    input_scaled = scaler.transform(input_array)
    z_pred = model.predict(input_scaled, verbose=False)
    st.success(f"✅ Gas Compressibility Z Factor estimated: {z_pred[0][0]}")
    print(z_pred)