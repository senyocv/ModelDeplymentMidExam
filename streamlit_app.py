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

# USER INPUT
st.sidebar.subheader("Input Data for Prediction")
user_input = {}

# Check numerical columns
numerical_columns = raw_data.drop(columns=cat_cols + ['booking_status']).select_dtypes(include=np.number).columns
st.write(f"Numerical Columns: {numerical_columns}")

# Debug: check if all are numeric
for col in numerical_columns:
    if not pd.api.types.is_numeric_dtype(raw_data[col]):
        st.warning(f"Column '{col}' is not numeric!")
        continue

    min_val = float(raw_data[col].min())
    max_val = float(raw_data[col].max())
    mean_val = float(raw_data[col].mean())

    if min_val == max_val:
        st.sidebar.warning(f"Warning: Column '{col}' has the same min and max value. Using default range.")
        min_val, max_val = 0, 10

    step = 1 if col != 'avg_price_per_room' else 0.1

    # Debug: Log each slider creation
    try:
        user_input[col] = st.sidebar.slider(
            col,
            min_val,
            max_val,
            mean_val,
            step=step
        )
    except Exception as e:
        st.error(f"Error with column '{col}': {e}")

# SELECT BOX
for col in cat_cols:
    raw_data[col] = raw_data[col].fillna("Unknown").astype(str)
for col in cat_cols:
    user_input[col] = st.sidebar.selectbox(col, sorted(raw_data[col].unique()))

# ppp debug check features
print("Expected feature names:")
print(model.feature_names_in_)

input_df = pd.DataFrame([user_input])
enc_arr = encoder.transform(input_df[cat_cols])
enc_df = pd.DataFrame(enc_arr, columns=encoder.get_feature_names_out(cat_cols))

input_df = input_df.drop(columns=cat_cols).reset_index(drop=True)
enc_df = enc_df.reset_index(drop=True)

final_input = pd.concat([input_df, enc_df], axis=1)

# ppp debug check final cols
print("Final input columns:")
print(final_input.columns)

expected_columns = model.feature_names_in_
final_input = final_input[expected_columns]

# Display user input data
st.subheader("User Input Data")
st.write(final_input)

# predict!!
if st.button("Predict"):
    prediction = model.predict(final_input)
    prediction_label = "Canceled" if prediction[0] == 1 else "Not Canceled"
    st.subheader("Prediction Result")
    st.write(f"Predicted Booking Status: **{prediction_label}**")
