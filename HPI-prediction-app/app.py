import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---- 1. Load the trained model ----
@st.cache_data
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# ---- 2. Define all feature names ----
FEATURE_NAMES = [
    'median_sale_price', 'median_list_price', 'homes_sold', 'new_listings',
    'inventory', 'median_dom', 'price', 'year', 'month', 'quarter',
    'hpi_lag_1', 'median_sale_price_lag_1', 'inventory_lag_1', 'hpi_lag_2',
    'median_sale_price_lag_2', 'inventory_lag_2', 'hpi_lag_3',
    'median_sale_price_lag_3', 'inventory_lag_3', 'hpi_lag_6',
    'median_sale_price_lag_6', 'inventory_lag_6', 'hpi_lag_12',
    'median_sale_price_lag_12', 'inventory_lag_12', 'sale_price_roll3',
    'hpi_diff_1'
]

# ---- 3. Set median/default values for all features ----
# You can replace these 0s with actual medians from training data
DEFAULT_FEATURES = np.zeros(len(FEATURE_NAMES))

# ---- 4. Streamlit App ----
st.title("Housing Price Index (HPI) Prediction")
st.write("Enter values for key features below. Other features are set to default values.")

# ---- 5. User inputs for key features ----
median_sale_price = st.number_input("Median Sale Price", value=300000.0)
median_list_price = st.number_input("Median List Price", value=320000.0)
homes_sold = st.number_input("Homes Sold", value=100)
new_listings = st.number_input("New Listings", value=50)
inventory = st.number_input("Inventory", value=200)

# ---- NEW: Inputs for month and year ----
year = st.number_input("Year", min_value=2000, max_value=2100, value=2025, step=1)
month = st.number_input("Month", min_value=1, max_value=12, value=11, step=1)

# ---- 6. Prepare input array ----
X_input = DEFAULT_FEATURES.copy()
X_input[FEATURE_NAMES.index('median_sale_price')] = median_sale_price
X_input[FEATURE_NAMES.index('median_list_price')] = median_list_price
X_input[FEATURE_NAMES.index('homes_sold')] = homes_sold
X_input[FEATURE_NAMES.index('new_listings')] = new_listings
X_input[FEATURE_NAMES.index('inventory')] = inventory
X_input[FEATURE_NAMES.index('year')] = year
X_input[FEATURE_NAMES.index('month')] = month

X_input = X_input.reshape(1, -1)

# ---- 7. Predict button ----
if st.button("Predict HPI"):
    try:
        hpi_prediction = model.predict(X_input)[0]
        st.success(f"Predicted HPI: {hpi_prediction:.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.write("---")
st.write("All other features are filled with default values to match the model's expected input.")
