import streamlit as st
import numpy as np
import pandas as pd
import joblib



import os

# ✅ Define base directory (works locally and in Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath('Customer Segmentation'))

# ✅ Define file paths
model_path = os.path.join(BASE_DIR, "Customer Segmentation", "Kmeans_model.pkl")
scaler_path = os.path.join(BASE_DIR, "Customer Segmentation", "scaler.pkl")

# ✅ Load files safely
with open(model_path, "rb") as f:
    kmeans = joblib.load(f)

with open(scaler_path, "rb") as f:
    scaler = joblib.load(f)



st.title('Customer Segmentation App')
st.write('Enter customer details to predict the segment.')

age = st.number_input('Age',min_value = 18, max_value = 100, value = 35)
income = st.number_input('Income',min_value = 0,max_value = 200000, value = 50000)
total_spending = st.number_input('Total Spending (sum of purchases)',min_value = 0, max_value = 5000, value = 1000 )
num_web_purchases = st.number_input('Number of web purchases', min_value = 0,max_value = 100, value = 10)
num_store_purchases = st.number_input('Number of Store Purchases',min_value = 0, max_value = 100, value = 10) 
num_web_visits = st.number_input('Nustreamlit runmber of Web Visits per month',min_value = 0,max_value = 50, value = 3) 
recency= st.number_input('Recency( days since last purchase)',min_value = 0, max_value = 365,value = 30) 


input_data = pd.DataFrame({
    'Age':[age],
    'Income':[income],
    'Total_spending':[total_spending],
    'NumWebPurchases':[num_web_purchases],
    'NumStorePurchases':[num_store_purchases],
    'NumWebVisitsMonth':[num_web_visits],
    'Recency':[recency]
})

input_scaled = scaler.transform(input_data)

if st.button('Predict Segment'):
    
    cluster = kmeans.predict(input_scaled)[0]
    
    st.success(f"Predicted Segment Cluster {cluster}")
    
    
    st.write("""
             Cluster 0 : High budget, web visiters
             Cluster 1 : High spending
             cluster 2 : Web visitors
             """)