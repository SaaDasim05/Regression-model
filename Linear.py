import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set up the Streamlit app
def main():
    st.title("Linear Regression Streamlit App")

    # Upload dataset
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("## Dataset Preview")
        st.write(data.head())

        # Feature selection
        st.sidebar.header("Select Features")
        features = st.sidebar.multiselect("Select independent variables (X)", options=data.columns)
        target = st.sidebar.selectbox("Select dependent variable (Y)", options=data.columns)

        if features and target:
            X = data[features]
            y = data[target]

            # Train-test split
            test_size = st.sidebar.slider("Test set size", 0.1, 0.5, 0.2)
            random_state = st.sidebar.number_input("Random state", value=42, step=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Train Linear Regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predictions and evaluation
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write("## Model Evaluation")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"R-squared (R2): {r2:.2f}")

            # Visualization
            st.write("## Prediction vs Actual Plot")
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Prediction vs Actual")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
