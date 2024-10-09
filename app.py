import numpy as np
import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
# Load the trained model
model = tf.keras.models.load_model('my_model.h5')

# Load the input scaler
with open('scaler_X.pkl', 'rb') as file_X:
    scaler_X = pickle.load(file_X)

# Load the output scaler
with open('scaler_Y.pkl', 'rb') as file_Y:
    scaler_Y = pickle.load(file_Y)

# Streamlit app
st.title('Beam Size Prediction')

# User input

rebar_ratio = st.number_input('rh0_s')
concrete_compressive_strength = st.number_input('fc_MPa')
rebar_yield_strength = st.number_input('fy_MPa')
Moment_Mn = st.number_input('M_kNm')
steel_strain = st.number_input('es')
neutralAxis_depth_ratio = st.number_input('c_d')
beam_width_to_depth_ratio = st.number_input('b_d')


input_data = pd.DataFrame({
    'rh0_s': [rebar_ratio],  
    'fc_MPa': [concrete_compressive_strength],
    'fy_MPa': [rebar_yield_strength],
    'M_kNm': [Moment_Mn],
    'es': [steel_strain],
    'c_d': [neutralAxis_depth_ratio],
    'b_d': [beam_width_to_depth_ratio]
})

input_data_scaled = scaler_X.transform(input_data)

# Make prediction (this gives scaled output)
prediction = model.predict(input_data_scaled)
prediction_scaled = scaler_Y.inverse_transform(prediction)

# Extract the beam width and depth from the array
predicted_width = prediction_scaled[0][0]  # First element for beam width
predicted_depth = prediction_scaled[0][1]  # Second element for beam depth

# Display the predicted beam size with proper formatting
st.write(f'Predicted Cross Section size for RC beam is: Width = {predicted_width:.2f} mm, Depth = {predicted_depth:.2f} mm')




