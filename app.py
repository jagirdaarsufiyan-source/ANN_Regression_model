# app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load Model and Scaler
model = load_model("model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

st.title("🏠 House Price Prediction using ANN")

st.write("Enter housing details:")

# Input fields
MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict Price"):
    
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)
    
    st.success(f"Predicted House Price: {prediction[0][0]:.2f} (in $100,000s)")