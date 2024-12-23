import streamlit as st
from sklearn.linear_model import LinearRegression
import numpy as np

# Define the features and the target variable
X = np.array([
    [1000, 3, 1, 1, 2],
    [1500, 4, 0, 1, 1],
    [900, 2, 1, 0, 1],
    [2000, 5, 1, 1, 3]
])
y = np.array([300000, 450000, 200000, 600000])  # House prices

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit interface
st.title("House Price Prediction")

# User inputs for house features
area = st.number_input("House Area (in square feet)", min_value=1, max_value=10000, value=1000)
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
water_heating = st.radio("Water Heating (1 for Yes, 0 for No)", options=[0, 1], index=0)
air_conditioning = st.radio("Air Conditioning (1 for Yes, 0 for No)", options=[0, 1], index=0)
car_parking = st.number_input("Number of Car Parking Spaces", min_value=0, max_value=10, value=2)

# Predict the house price
user_features = np.array([[area, rooms, water_heating, air_conditioning, car_parking]])
predicted_price = model.predict(user_features)

# Display the result
st.write(f"The predicted house price is: ${predicted_price[0]:,.2f}")
