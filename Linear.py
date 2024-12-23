import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Streamlit app
st.title("House Price Prediction")
st.write("This app predicts house prices based on user inputs.")

# Define input fields
st.write("## Enter House Features")
area = st.number_input("House Area (in square feet):", min_value=0.0, step=1.0)
rooms = st.number_input("Number of Rooms:", min_value=0, step=1)
water_heating = st.selectbox("Water Heating System:", options=["Yes", "No"])
air_conditioning = st.selectbox("Air Conditioning:", options=["Yes", "No"])
car_parking = st.number_input("Car Parking Spaces:", min_value=0, step=1)

# Convert categorical inputs to numerical
water_heating_value = 1 if water_heating == "Yes" else 0
air_conditioning_value = 1 if air_conditioning == "Yes" else 0

# Create input dataframe
input_data = pd.DataFrame({
    "Area": [area],
    "Rooms": [rooms],
    "WaterHeating": [water_heating_value],
    "AirConditioning": [air_conditioning_value],
    "CarParking": [car_parking]
})

# Placeholder for linear regression model coefficients (pretrained values)
st.write("## Note: Using pre-trained model coefficients for demonstration.")

# Example pre-trained coefficients
coefficients = {
    "Area": 150,
    "Rooms": 20000,
    "WaterHeating": 5000,
    "AirConditioning": 10000,
    "CarParking": 3000
}
intercept = 50000

# Predict house price
price = (
    coefficients["Area"] * input_data["Area"].iloc[0] +
    coefficients["Rooms"] * input_data["Rooms"].iloc[0] +
    coefficients["WaterHeating"] * input_data["WaterHeating"].iloc[0] +
    coefficients["AirConditioning"] * input_data["AirConditioning"].iloc[0] +
    coefficients["CarParking"] * input_data["CarParking"].iloc[0] +
    intercept
)

# Display the predicted price
st.write("## Predicted House Price")
st.write(f"$ {price:,.2f}")
