import pickle
import numpy as np
import streamlit as st

model = pickle.load(open("model.sav", "rb"))
scaler = pickle.load(open("scaler1.sav", "rb"))
lb = pickle.load(open('label_cloudcover.sav','rb'))
lb1 = pickle.load(open('label_season.sav','rb'))
lb2 = pickle.load(open('label_location.sav','rb'))
lb3 = pickle.load(open('label_weathertype.sav','rb'))


st.title("🌦 Weather Prediction App")

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
wind_speed = st.number_input("Wind Speed")
precipitation = st.number_input("Precipitation (%)")
cloud = st.selectbox("Cloud Cover", lb.classes_)
pressure = st.number_input("Atmospheric Pressure")
uv = st.number_input("UV Index")
season = st.selectbox("Season", lb1.classes_)
visibility = st.number_input("Visibility (km)")
location = st.selectbox("Location", lb2.classes_)
if st.button("Predict"):

    cloud_num = lb.transform([cloud])[0]
    season_num = lb1.transform([season])[0]
    location_num = lb2.transform([location])[0]

    input_data = [[
        temperature, humidity, wind_speed, precipitation,
        cloud_num, pressure, uv, season_num, visibility, location_num
    ]]

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)

    weather_label = lb3.inverse_transform(pred)[0]

    st.success(f"🌤 Weather Type: **{weather_label}**")
