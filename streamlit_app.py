import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

@st.cache_resource
def load_model():
    with open("rf_md_uts.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoder():
    df = pd.read_csv("Dataset_B_hotel.csv")
    st.write("Columns in dataset:", df.columns.tolist())
    cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    
    df[cat_cols] = df[cat_cols].fillna("Unknown")
    
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoder.fit(df[cat_cols])
    return encoder




model = load_model()
encoder = load_encoder()
cat_cols = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
raw_data = pd.read_csv("Dataset_B_hotel.csv")



st.sidebar.title("Hotel Booking Cancellation Predictor")

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(raw_data)

# USER INPUT
st.sidebar.subheader("Input Data for Prediction")
user_input = {}

# AUTO SLIDER
numerical_columns = raw_data.drop(columns=cat_cols + ['booking_status']).select_dtypes(include=np.number).columns
for col in numerical_columns:
    user_input[col] = st.sidebar.slider(
        col,
        float(raw_data[col].min()),
        float(raw_data[col].max()),
        float(raw_data[col].mean())
    )

# SELECT BOX
for col in cat_cols:
    raw_data[col] = raw_data[col].fillna("Unknown").astype(str)
for col in cat_cols:
    user_input[col] = st.sidebar.selectbox(col, sorted(raw_data[col].unique()))

# gas pred
input_df = pd.DataFrame([user_input])
enc_arr = encoder.transform(input_df[cat_cols])
enc_df = pd.DataFrame(enc_arr, columns=encoder.get_feature_names_out(cat_cols))

input_df = input_df.drop(columns=cat_cols).reset_index(drop=True)
enc_df = enc_df.reset_index(drop=True)
final_input = pd.concat([input_df, enc_df], axis=1)

st.subheader("User Input Data")
st.write(final_input)

if st.button("Predict"):
    prediction = model.predict(final_input)
    prediction_label = "Canceled" if prediction[0] == 1 else "Not Canceled"
    st.subheader("Prediction Result")
    st.write(f"Predicted Booking Status: **{prediction_label}**")
