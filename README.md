# 🧠 Predict Z-Factor App

This Streamlit web application uses a TensorFlow-trained neural network to predict the gas compressibility factor (Z-factor) based on:

- Pressure (psia)
- Temperature (°R)
- Specific Gravity

It also uses a `StandardScaler` from Scikit-learn for data normalization.

## 🛠 How to Use

1. Input pressure, temperature, and specific gravity.
2. Click the **Predict Z-Factor** button.
3. The predicted Z-factor will be displayed below.

## 📁 Files

- `src/app.py`: Main Streamlit application
- `model/z_model_2025_04_15_v1.keras`: Trained Keras model
- `model/scaler_z_2025_04_15.pkl`: Scaler used for data preprocessing
- `data/...`: Values are determining in this file
- `requirements.txt`: Python dependencies
- `.huggingface.yml`: Hugging Face deployment configuration

## 🚀 Deployment

This app is ready to be deployed on [Hugging Face Spaces](https://huggingface.co/spaces).

---
