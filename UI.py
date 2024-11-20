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
import matplotlib.pyplot as plt
# Title of the App
st.markdown("<h1 style='text-align: center; color: black;'>ETF Portfolio Optimization</h1>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.markdown("<h2 style='color: black;'>User Input</h2>", unsafe_allow_html=True)
risk_tolerance = st.sidebar.selectbox(
    "Select Risk Tolerance Level:",
    ("Low Risk", "Medium Risk", "High Risk")
)
# Medium Risk Options
if risk_tolerance == "Medium Risk":
    st.markdown("<h3 style='color: black;'>Medium Risk Options</h3>", unsafe_allow_html=True)
    exclude_preferred_share = st.checkbox("Exclude Preferred Share ETFs")
    exclude_reit = st.checkbox("Exclude REIT ETFs")
    excluded_types = []
    if exclude_preferred_share:
        excluded_types.append("Preferred Share")
    if exclude_reit:
        excluded_types.append("REIT")

# High Risk Options
if risk_tolerance == "High Risk":
    st.markdown("<h3 style='color: black;'>High Risk Options</h3>", unsafe_allow_html=True)

    # Are you a Day Trader?
    st.markdown("<h4 style='color: black;'>Are you a Day Trader?</h4>", unsafe_allow_html=True)
    day_trader = st.radio("Day Trader:", ["Yes", "No"], key="day_trader_key")

    if day_trader == "No":
        # Include Cryptocurrency ETFs?
        st.markdown("<h4 style='color: black;'>Do you want to include Cryptocurrency ETFs?</h4>", unsafe_allow_html=True)
        include_crypto = st.radio("Include Cryptocurrency ETFs:", ["Yes", "No"], key="crypto_key")
# Load data based on risk tolerance
data = None
if risk_tolerance == "Low Risk":
    data_path = "historical_cumulative_returns_lr.csv"
    data = pd.read_csv(data_path, index_col="Date", parse_dates=True)
elif risk_tolerance == "Medium Risk":
    data_path = "historical_cumulative_returns_mr_all.csv"  # Adjust this path as necessary
    data = pd.read_csv(data_path, index_col="Date", parse_dates=True)
elif risk_tolerance == "High Risk":
    if day_trader == "Yes":
        data_path = "historical_cumulative_returns_hr_day.csv"
    elif day_trader == "No" and include_crypto == "Yes":
        data_path = "historical_cumulative_returns_hr_noday.csv"
    else:
        data_path = "historical_cumulative_returns_hr_nocry.csv"
    data = pd.read_csv(data_path, index_col="Date", parse_dates=True)
# Key Statistics and Visualization Options
st.markdown("<h2 style='color: black;'>Analysis and Visualization</h2>", unsafe_allow_html=True)

# 选项供用户选择如何显示内容
show_option = st.radio(
    "How would you like to display key performance and cumulative returns?",
    ("Show Key Statistics Only", "Show Cumulative Returns Only", "Show Portfolio Weights Only", "Show All"),
    index=0  # 默认选择 "Show Key Statistics Only"
)

# 加载对应的统计信息数据文件
stats = None
if data is not None:
    if risk_tolerance == "Low Risk":
        stats_path = "key_performance_statistics_with_weights_lr.csv"
    elif risk_tolerance == "Medium Risk":
        stats_path = "key_performance_statistics_with_weights_mr_all.csv"
    elif risk_tolerance == "High Risk":
        if day_trader == "Yes":
            stats_path = "key_performance_statistics_with_weights_hr_day.csv"
        elif day_trader == "No" and include_crypto == "Yes":
            stats_path = "key_performance_statistics_with_weights_hr_noday.csv"
        elif day_trader == "No" and include_crypto == "No":
            stats_path = "key_performance_statistics_with_weights_hr_nocry.csv"
    
    # 读取统计信息文件
    stats = pd.read_csv(stats_path)

# 提取关键统计信息
if stats is not None:
    expected_annual_return = stats.iloc[0, 1]  # 假设第一行第二列存储期望年收益
    annual_volatility = stats.iloc[1, 1]       # 第二行第二列存储波动率
    sharpe_ratio = stats.iloc[2, 1]            # 第三行第二列存储夏普比率
portfolio_weights = stats.iloc[3:].dropna(how="all")  # 移除完全为空的行
portfolio_weights.reset_index(drop=True, inplace=True)
if 'ETF' in portfolio_weights.columns and 'Weight' in portfolio_weights.columns:
    portfolio_weights = portfolio_weights[['ETF', 'Weight']]
else:
    portfolio_weights = None
    
# 绘制累计收益图
cumulative_returns_plot = None
if data is not None:
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(ax=ax, title="Cumulative Returns Over Time")
    ax.set_ylabel("Cumulative Returns")
    ax.set_xlabel("Date")
    cumulative_returns_plot = fig

# 根据用户选择展示内容
if show_option == "Show Key Statistics Only":
    # 仅显示关键统计信息
    st.subheader("Key Performance Statistics")
    st.markdown(f"**Expected Annual Return:** {expected_annual_return}")
    st.markdown(f"**Annual Volatility:** {annual_volatility}")
    st.markdown(f"**Sharpe Ratio:** {sharpe_ratio}")

elif show_option == "Show Cumulative Returns Only":
    # 仅显示累计收益图
    st.subheader("Cumulative Returns")
    if cumulative_returns_plot:
        st.pyplot(cumulative_returns_plot)
    else:
        st.error("Cumulative returns plot could not be generated due to invalid data.")

elif show_option == "Show Portfolio Weights Only":
    # 仅显示 Portfolio Weights
    st.subheader("Portfolio Weights")
    if portfolio_weights is not None:
        for index, row in portfolio_weights.iterrows():
            st.markdown(f"<strong>{row['ETF']}:</strong> {row['Weight']}", unsafe_allow_html=True)
    else:
        st.error("Portfolio weights data could not be loaded or is incorrectly formatted.")
else:
    # 同时显示所有内容
    st.subheader("Key Performance Statistics")
    st.markdown(f"**Expected Annual Return:** {expected_annual_return}")
    st.markdown(f"**Annual Volatility:** {annual_volatility}")
    st.markdown(f"**Sharpe Ratio:** {sharpe_ratio}")
    
    st.subheader("Cumulative Returns")
    if cumulative_returns_plot:
        st.pyplot(cumulative_returns_plot)
    else:
        st.error("Cumulative returns plot could not be generated due to invalid data.")
    
    st.subheader("Portfolio Weights")
    if portfolio_weights is not None:
        for index, row in portfolio_weights.iterrows():
            st.markdown(f"<strong>{row['ETF']}:</strong> {row['Weight']}", unsafe_allow_html=True)
    else:
        st.error("Portfolio weights data could not be loaded or is incorrectly formatted.")

# 引入数据文件：明确引用你的路径


    # 加载 Portfolio Weights 数据
    portfolio_weights = stats.iloc[3:].dropna(how="all")  # 假设 Portfolio Weights 从 stats 文件第4行开始
    portfolio_weights.reset_index(drop=True, inplace=True)

    # 确保 Portfolio Weights 数据正确
    if 'ETF' in portfolio_weights.columns and 'Weight' in portfolio_weights.columns:
        portfolio_weights = portfolio_weights[['ETF', 'Weight']]
        portfolio_weights["Weight"] = portfolio_weights["Weight"].str.rstrip('%').astype(float) / 100
    else:
        st.error("The portfolio weights data format is incorrect. Please check the input file.")
        portfolio_weights = None

# Streamlit Sidebar Inputs for Simulation
st.sidebar.markdown("<h2 style='color: black;'>Simulation Settings</h2>", unsafe_allow_html=True)


# Slider to select simulation years
simulation_years = st.sidebar.slider(
    "Select Simulation Years:",
    min_value=1,
    max_value=10,
    value=5,  # Default value
    step=1
)

# Slider to select the number of simulation paths
simulations = st.sidebar.slider(
    "Select Number of Simulation Paths:",
    min_value=100,
    max_value=5000,
    value=1000,  # Default value
    step=100
)

# Simulation Function: GBM
def simulate_portfolio_gbm(data, portfolio_weights, years, num_simulations):
    """
    Simulates portfolio returns using Geometric Brownian Motion (GBM).

    Args:
        data (pd.DataFrame): Historical cumulative return data.
        portfolio_weights (pd.DataFrame): Portfolio weights.
        years (int): Number of years to simulate.
        num_simulations (int): Number of Monte Carlo simulation paths.

    Returns:
        pd.DataFrame: Simulated portfolio returns.
    """
    if data is None or portfolio_weights is None:
        st.error("Historical data or portfolio weights are missing. Unable to simulate returns.")
        return None

    # Align historical data with portfolio weights
    historical_data = data
    historical_data.columns = ["Cumulative Return"]  # Ensure proper column name

    # Calculate historical daily returns
    daily_returns = historical_data.pct_change().dropna()
    mu = daily_returns.mean().values[0]  # Mean daily return
    sigma = daily_returns.std().values[0]  # Daily volatility

    # Simulation parameters
    num_days = years * 252  # Number of trading days
    last_price = historical_data.iloc[-1, 0]  # Last historical cumulative return
    dt = 1 / 252  # Time step (daily)

    # Simulate GBM paths
    simulated_paths = np.zeros((num_days, num_simulations))
    simulated_paths[0] = last_price  # Start at last price

    for t in range(1, num_days):
        random_shocks = np.random.normal(0, 1, num_simulations)
        simulated_paths[t] = (
            simulated_paths[t - 1] *
            np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks)
        )

    # Convert simulated paths to DataFrame
    simulated_df = pd.DataFrame(
        simulated_paths,
        index=pd.date_range(start=historical_data.index[-1], periods=num_days, freq="B"),
        columns=[f"Simulation {i+1}" for i in range(num_simulations)]
    )
    return simulated_df

# Load portfolio weights
portfolio_weights = None
if stats is not None:
    portfolio_weights = stats.iloc[3:].dropna(how="all")[["ETF", "Weight"]]

# Perform Simulation
if data is not None and portfolio_weights is not None:
    st.subheader("Simulation Results")
    
    simulated_returns = simulate_portfolio_gbm(data, portfolio_weights, simulation_years, simulations)

    
    if simulated_returns is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        simulated_returns.iloc[:, :10].plot(ax=ax, alpha=0.7)  # Show 10 paths
        ax.set_title("Simulated Portfolio Cumulative Returns (GBM)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value")
        st.pyplot(fig)

        # Calculate Simulated Metrics
        daily_simulated_returns = simulated_returns.pct_change().mean(axis=1).dropna()
        expected_simulated_return = (1 + daily_simulated_returns.mean()) ** 252 - 1
        simulated_volatility = daily_simulated_returns.std() * np.sqrt(252)
    else:
        st.error("Simulation could not be performed due to missing or invalid data.")
else:
    st.error("Historical data or portfolio weights are missing. Unable to simulate returns.")







