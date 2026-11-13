import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
from pathlib import Path


st.set_page_config(
    page_title="Bank Customer Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Get the directory containing this application
BASE_DIR = Path(__file__).resolve().parent

# Load the salary regression ANN model
model = tf.keras.models.load_model(
    str(BASE_DIR / "salary_model.h5")
)

# Load salary regression encoders
with open(BASE_DIR / "label_encoder_gender_salary.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open(BASE_DIR / "onehot_encoder_geo_salary.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

# Load salary regression scaler
with open(BASE_DIR / "scaler_salary.pkl", "rb") as f:
    scaler = pickle.load(f)

## streamlit app

st.title("💰 Bank Customer Salary Prediction")

st.markdown(
    "Estimate a bank customer's salary using an "
    "Artificial Neural Network (ANN) regression model."
)

st.divider()



# User input

st.subheader("Customer Information")

col1, col2 = st.columns(2, gap="large")

with col1:
    geography = st.selectbox(
        "Geography", onehot_encoder_geo.categories_[0]
    )
    gender = st.selectbox(
        "Gender", label_encoder_gender.classes_
    )
    age = st.slider("Age", 18, 92, 35)
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=650,
        step=1
    )
    tenure = st.slider("Tenure", 0, 10, 5)

with col2:
    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )
    num_of_products = st.slider(
        "Number of Products", 1, 4, 2
    )
    has_cr_card = st.selectbox(
        "Has Credit Card", [0, 1]
    )
    is_active_member = st.selectbox(
        "Is Active Member", [0, 1]
    )
    Exited = st.selectbox(
        "Customer Exited", [0, 1]
    )


# Preprocess the input data
# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geography_encoded = onehot_encoder_geo.transform([[geography]])[0]

# Scale the input data
input_data = np.array([[credit_score, gender_encoded, age, tenure, balance, num_of_products, has_cr_card, is_active_member, Exited,*geography_encoded]])
input_data = scaler.transform(input_data)


# Predict salary only when the button is clicked
if st.button("Predict Estimated Salary", type="primary"):
    prediction = model.predict(input_data, verbose=0)
    predicted_salary = float(prediction[0][0])

    st.subheader("Prediction Result")
    st.metric(
        "Predicted Estimated Salary",
        f"{predicted_salary:,.2f}"
    )

