import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pickle
from pathlib import Path



st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Get the folder containing app.py
BASE_DIR = Path(__file__).resolve().parent

# Load the trained ANN model
model = tf.keras.models.load_model(
    str(BASE_DIR / "model.h5")
)

# Load the encoders and scaler
with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open(BASE_DIR / "onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open(BASE_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

## streamlit app

st.title("🏦 Bank Customer Churn Prediction")
st.markdown(
    "Predict whether a bank customer is likely to leave "
    "using an Artificial Neural Network (ANN)."
)
st.divider()


## user input
# age = st.number_input("Age", min_value=0, max_value=120, value=30)
# gender = st.selectbox("Gender", options=["Male", "Female"])
# geography = st.selectbox("Geography", options=["France", "Spain", "Germany"])
# balance = st.number_input("Balance", min_value=0.0, value=0.0)
# credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
# tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
# has_cr_card = st.checkbox("Has Credit Card")
# is_active_member = st.checkbox("Is Active Member")

# user input

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
        "Credit Score", min_value=300,
        max_value=850, value=650, step=1
    )
    tenure = st.slider("Tenure", 0, 10, 5)

with col2:
    balance = st.number_input(
        "Account Balance", min_value=0.0,
        value=50000.0, step=1000.0
    )
    estimated_salary = st.number_input(
        "Estimated Salary", min_value=0.0,
        max_value=200000.0, value=50000.0,
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


## prepare the input data for prediction only for multiple samples
# input_data = pd.DataFrame( {
#     'CreditScore': [credit_score],
#     'Gender': [label_encoder_gender.transform([gender])[0]],
#     'Age': [age],
#     'Tenure': [tenure],
#     'Balance': [balance],
#     'Tenure': [tenure],
#     'NumOfProducts': [num_of_products],
#     'HasCrCard': [has_cr_card],
#     'IsActiveMember': [is_active_member]
#     'EstimatedSalary': [estimated_salary],
# } )

## one-hot encode the geography column
# geography_encoded = onehot_encoder_geo.transform([[geography]])
#geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Concatenate the input data with the one-hot encoded geography
# input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)



# Preprocess the input data
# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# One-hot encode geography
geography_encoded = onehot_encoder_geo.transform([[geography]])[0]

# Scale the input data
input_data = np.array([[credit_score, gender_encoded, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary,*geography_encoded]])
input_data = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data)
# st.write("Prediction probability:", prediction[0][0])


# Predict customer churn only when the button is clicked
if st.button("Predict Customer Churn", type="primary"):

    prediction = model.predict(input_data, verbose=0)
    churn_probability = float(prediction[0][0])

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    if churn_probability > 0.5:
        st.error("The customer is likely to churn.")
    else:
        st.success("The customer is unlikely to churn.")
