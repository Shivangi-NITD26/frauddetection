import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
from datetime import datetime

@st.cache_resource
def Initialize_models():
    models = {
        "Logistic Regression": pk.load(open("logistic_regression.pkl", "rb")),
        "Decision Tree": pk.load(open("decision_tree.pkl", "rb"))
    }
    scaler = pk.load(open("scaler.pkl", "rb"))
    freq_maps = pk.load(open("freq_maps.pkl", "rb"))
    return models, scaler, freq_maps

models, scaler, freq_maps = Initialize_models()

st.title("CREDIT CARD FRAUD DETECTOR")

selected_model = st.radio(
    "Choose a model to run the prediction:",
    list(models.keys()),
    index=0,
    horizontal=True
)

st.write("""
Welcome!  
This tool helps you find out if a credit card transaction might be fraud.  
Just enter the details below and click **Predict** to see the result.  
Note: This tool is designed for transactions within the United States.
""")

with st.form("transaction_form"):
    st.header("Personal Information")
    first = st.text_input("First Name", "Shivangi")
    last = st.text_input("Last Name", "Patwa")
    dob = st.date_input("Date of Birth", datetime(2004, 9, 9))
    gender = st.selectbox("Gender", ["F", "M"])
    job = st.text_input("Occupation", "Engineer")
    age = datetime.now().year - dob.year

    st.header("Location Information")
    street = st.text_input("Street Address", "123 Main St")
    city = st.text_input("City", "")
    city_pop = st.number_input("City Population", min_value=0, value=1000000)
    state = st.selectbox("State", [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
        "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
        "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
        "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
        "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
        "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
    ])
    zip_code = st.text_input("ZIP Code", "10001")
    lat = st.number_input("Latitude", value=40.7128)
    long = st.number_input("Longitude", value=-74.0060)

    st.header("Transaction Details")
    trans_date = st.date_input("Transaction Date", datetime.now())
    trans_time = st.time_input("Transaction Time", datetime.now().time())
    amt = st.number_input("Amount ($)", min_value=0.0, value=100.0, step=1.0)
    cc_num = st.text_input("Credit Card Number", "1234567890123456")
    category = st.selectbox("Category", sorted(freq_maps['category'].index))
    merchant = st.text_input("Merchant", "fraud_Kirlin and Sons")
    merch_lat = st.number_input("Merchant Latitude", value=40.7128)
    merch_long = st.number_input("Merchant Longitude", value=-74.0060)

    trans_datetime = datetime.combine(trans_date, trans_time)
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    input_data = {
        'trans_date_trans_time': [trans_datetime],
        'cc_num': [int(cc_num)],
        'amt': [amt],
        'first': [first],
        'last': [last],
        'gender': [gender],
        'street': [street],
        'city': [city],
        'state': [state],
        'zip': [zip_code],
        'city_pop': [city_pop],
        'job': [job],
        'dob': [dob],
        'merchant': [merchant],
        'category': [category],
        'lat': [lat],
        'long': [long],
        'merch_lat': [merch_lat],
        'merch_long': [merch_long]
    }

    df = pd.DataFrame(input_data)

    cols_to_drop = ['trans_num', 'unix_time', 'Unnamed: 0', 'first', 'last', 'street']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])

    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['weekday'] = df['trans_date_trans_time'].dt.weekday
    df['month'] = df['trans_date_trans_time'].dt.month
    df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

    df.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)

    df['gender_M'] = (df['gender'] == 'M').astype(int)
    df.drop(columns=['gender'], inplace=True)

    for col in ['city', 'state', 'job', 'merchant', 'category', 'zip']:
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(freq_maps.get(col, {})).fillna(0)
            df.drop(columns=[col], inplace=True)

    expected_columns = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else [
        'cc_num', 'amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long',
        'gender_M', 'hour', 'day', 'weekday', 'month', 'age',
        'city_encoded', 'state_encoded', 'job_encoded', 'merchant_encoded',
        'category_encoded', 'zip_encoded'
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    scaled_data = scaler.transform(df)

    st.subheader("Prediction Results")
    model = models[selected_model]
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]

    if prediction == 1:
        st.error(f"Fraud Detected (Fraud Probability: {proba:.2%})")
    else:
        st.success(f"Legitimate Transaction (Fraud Probability: {proba:.2%})")
