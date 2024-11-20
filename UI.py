# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 12:49:44 2024

@author: qusha
"""

import streamlit as st
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# Title of the App
st.title("ETF Portfolio Optimization")
st.sidebar.header("User Input")
risk_tolerance = st.sidebar.selectbox(
    "Select Risk Tolerance Level:",
    ("Low Risk", "Medium Risk", "High Risk")
)

# Additional options for Medium Risk customers
if risk_tolerance == "Medium Risk":
    st.sidebar.subheader("Do you want to exclude ETF you are not familiar with?")
    exclude_preferred_share = st.sidebar.checkbox("Exclude Preferred Share ETFs")
    exclude_reit = st.sidebar.checkbox("Exclude REIT ETFs")
    excluded_types = []
    if exclude_preferred_share:
        excluded_types.append("Preferred Share")
    if exclude_reit:
        excluded_types.append("REIT")
if risk_tolerance == "High Risk":
    st.sidebar.subheader("High Risk Customization")
    day_trader = st.sidebar.radio("Are you a Day Trader?", ("Yes", "No"))
    
    include_crypto = None
    if day_trader == "No":
        include_crypto = st.sidebar.radio("Do you want to include Cryptocurrency ETFs?", ("Yes", "No"))

investment_amount = st.sidebar.number_input(
    "Investment Amount ($):",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000,
    help="Enter the amount you want to invest."
)
simulate_years = st.sidebar.slider(
    "Simulate Future Returns (Years):",
    min_value=1,
    max_value=10,
    value=5,
    help="Choose the number of years for future return simulation."
)


# Load data based on risk tolerance
data = None
if risk_tolerance == "Low Risk":
    data_path = "historical_cumulative_returns_lr.csv"
    data = pd.read_csv(data_path, index_col="Date", parse_dates=True)
elif risk_tolerance == "Medium Risk":
    # Mocked Medium Risk data loading with customization
    # Replace 'all_medium_risk_data' with your actual merged dataset
    all_medium_risk_data = {
        "Preferred Share": pd.DataFrame(),  # Replace with actual data
        "REIT": pd.DataFrame(),  # Replace with actual data
        "Others": pd.DataFrame(),  # Replace with all non-excluded data
    }

