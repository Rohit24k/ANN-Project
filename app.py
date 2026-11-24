import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle
import streamlit as st
import pandas as pd
import numpy as np

## Load the trained model
model = tf.keras.models.load_model('model.h5')

## Load scaler, ohe_geo and label_encoder file
with open('ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title('Customer Churn Prediction')

## User Input
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age',0, 80 )
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score' )
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active', [0,1])

## Input Data

input_data = pd.DataFrame( {
    'CreditScore': credit_score,
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
})

geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = ohe_geo.get_feature_names_out(['Geography']))

## One hot encoded with input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis = 1)

## Scale the Data

input_scaled = scaler.transform(input_data)

## Prediction churn

prediction = model.predict(input_scaled)
prediction_probab = prediction[0][0]

if prediction_probab > 0.5:
    st.write('Probability is ',prediction_probab)
else:
    st.write('Probability is ',prediction_probab)
if prediction_probab > 0.5:
    print('Not exit')
else:
    print('Exited')
