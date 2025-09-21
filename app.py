import streamlit as st
import pandas as pd
import pickle

st.title("🚗 Used Car Price Prediction...")
st.write("Give in the details of the used car to predict its price....")

# load model and encoders
with open('models/used_cars_prediction.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# load dataset for dropdown options
data = pd.read_csv('data/used_cars.csv')

# brand selection
brands = data['make'].unique()
brand = st.selectbox("Select Brand", brands)

# car model selection based on brand
models = data[data['make'] == brand]['model'].unique()
car_model = st.selectbox("Select Model", models)

# city selection
cities = data['city'].unique()
city = st.selectbox("Enter your City", cities)

mileage = st.slider("Enter Mileage (in KM)",min_value=0, max_value=150000, value=50000)
yr = st.slider("Enter year of manufacture",min_value=2000, max_value=2025, value=2020)
fuel = st.radio("Select Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'Hybrid'])
transmission = st.radio("Select Transmission Type", options=['Manual', 'automatic'])
owner = st.number_input("Enter Number of Previous Owners", min_value=0, max_value=10, value=1)

if st.button("Predict Price"):
    input_data = {
        'make': brand,
        'model': car_model,
        'city': city,
        'mileage': mileage,
        'make_year': yr,
        'fuel_type': fuel.lower(),
        'transmission': transmission.lower(),
        'no_of_owners': owner
    }
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in input_data:
            input_data[col] = le.transform([input_data[col]])[0]
    
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, drop_first=True)

    # Align the input data with the model's expected features
    model_features = model.feature_names_in_
    for feature in model_features:
        if feature not in df.columns:
            df[feature] = 0  # Add missing feature with default value 0
    df = df[model_features]  # Reorder columns to match model's features    

    prediction = model.predict(df)[0]
    st.success(f"The predicted price of the car is: ₹ {prediction:,.2f}")


