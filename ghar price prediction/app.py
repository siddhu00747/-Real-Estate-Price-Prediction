import pandas as pd
import numpy as np
import pickle
import streamlit as st

# model load krege
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('le.pkl', 'rb') as le_file:
    le = pickle.load(le_file)

# Streamlit UI
st.title("House Price Prediction")

# user se input lege
house_size = st.number_input("Enter House Size (in sqft)", min_value=500, max_value=5000, step=50)
bedrooms = st.number_input("Enter Number of Bedrooms", min_value=1, max_value=10)
bathrooms = st.number_input("Enter Number of Bathrooms", min_value=1, max_value=10)
toilet_area = st.number_input("Enter Toilet Area (in sqft)", min_value=30, max_value=300, step=10)
location = st.selectbox("Select Location", ["Urban", "Suburban", "Rural"])

# When the user clicks the 'Predict Price' button
if st.button("Predict Price"):
    #loc encode hoga
    location_encoded = le.transform([location])[0]

    # Create the feature array for prediction
    features = np.array([house_size, bedrooms, bathrooms, toilet_area, location_encoded]).reshape(1, -1)

    # Predict the price
    predicted_price = model.predict(features)[0]
    
    # Display the predicted price
    st.write(f"Predicted House Price: ${predicted_price:,.2f}")
