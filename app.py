import streamlit as st
import numpy as np
from catboost import CatBoostRegressor

st.set_page_config(layout="wide")

st.title("California Housing Price Prediction")

model = CatBoostRegressor()
model.load_model("catboost_california.cbm")

values = np.zeros(16)

col1, col2, col3 = st.columns([1,1,1])

with col1:
    longitude = st.number_input(max_value=-114.1315, min_value=-124.6509, label="Longitude")
    latitude = st.number_input(max_value=42.0126, min_value=32.5121, label="Latitude")
    population = st.number_input(max_value=35000, min_value=1, label="Population")

with col2:
    households = st.number_input(max_value=10000, min_value=1, label="Total Household")
    median_inc = st.number_input(max_value=150000, min_value=1, label="Median Income(In USD)")/10000
    median_age = st.number_input(max_value=52, min_value=1, label="Median House Age(In Years)")

with col3:
    total_rooms = st.number_input(min_value=1, max_value=40000, label="Total Rooms")
    total_bedrooms = st.number_input(min_value=1, max_value=6445, label="Total Bedrooms")

population_per_household = population/households
bedroom_per_room = total_bedrooms/total_rooms
room_per_household = total_rooms/households
income_sq = median_inc**2
income_log = np.log1p(median_inc)
lat_sq = latitude**2
long_sq = longitude**2
long_x_lat = longitude * latitude
population_per_household_log = np.log1p(population_per_household)

san_francisco_coordinates = (37.7749, -122.4194)
los_angeles_coordinates = (34.0549, -118.2426)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

distance_sf = haversine(latitude, longitude, san_francisco_coordinates[0], san_francisco_coordinates[1])
distance_la = haversine(latitude, longitude, los_angeles_coordinates[0], los_angeles_coordinates[1])

values[0], values[1], values[2] = longitude, latitude, median_age
values[3], values[4], values[5] = households, median_inc, room_per_household
values[6], values[7], values[8] = bedroom_per_room, population_per_household, income_sq
values[9], values[10], values[11] = income_log, lat_sq, long_sq
values[12], values[13], values[14] = long_x_lat, population_per_household_log, distance_sf
values[15] = distance_la


if st.button("Predict"):
    pred = model.predict(values)

    st.subheader(f"Prediction: Median House Value = ${pred}")
