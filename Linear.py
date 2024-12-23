import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title
st.title("Linear Regression Model App")

# Sidebar for user input
st.sidebar.header("Upload Your Dataset")
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if file is not None:
    # Load dataset
    data = pd.read_csv(file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Feature selection
    st.sidebar.header("Model Configuration")
    features = st.sidebar.multiselect("Select Feature Columns", options=data.columns)
    target = st.sidebar.selectbox("Select Target Column", options=data.columns)

    if features and target:
        X = data[features]
        y = data[target]

        # Split data
        test_size = st.sidebar.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100
        random_state = st.sidebar.number_input("Random State (Optional)", value=0, step=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Model training
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Model performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("### Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-squared (R2): {r2:.2f}")

        # Visualizations
        st.write("### Predictions vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Display coefficients
        st.write("### Model Coefficients")
        coefficients = pd.DataFrame({
            "Feature": features,
            "Coefficient": model.coef_
        })
        st.dataframe(coefficients)
    else:
        st.write("Please select at least one feature and a target variable.")
else:
    st.write("Please upload a CSV file to get started.")
