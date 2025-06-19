import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



# Function to calculate drawdown
def calculate_drawdown(cumulative_returns):
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown

# Function to load and process the CSV file
def process_data(file):
    df = pd.read_csv(file, delimiter=';')
    
    # Convert 'Entry time' to datetime
    df['Entry time'] = pd.to_datetime(df['Entry time'])
    
    # Filter data for the 8 months period starting from the most recent entry
    start_date = df['Entry time'].max() - pd.DateOffset(months=8)
    df = df[df['Entry time'] >= start_date]
    
    # Calculate daily PNL
    df['Daily PNL'] = df['PNL']  # Using PNL directly; adjust logic if needed
    
    # Calculate cumulative PNL
    df['Cumulative PNL'] = df['Daily PNL'].cumsum()
    
    # Calculate returns
    df['Daily Return'] = df['Daily PNL'] / 100000  # Adjust based on your starting balance
    df['Cumulative Return'] = (1 + df['Daily Return']).cumprod() - 1
    
    return df

# Streamlit UI
st.title("Financial Metrics Dashboard")
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

if uploaded_file is not None:
    df = process_data(uploaded_file)
    
    # Cumulative Returns Chart
    st.subheader("Cumulative Returns")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Entry time'], y=df['Cumulative Return'], mode='lines', name='Cumulative Return'))
    st.plotly_chart(fig)

    # Monthly Return Chart
    st.subheader("Monthly Return")
    df['Month'] = df['Entry time'].dt.to_period('M')
    monthly_returns = df.groupby('Month')['Daily Return'].sum()
    fig = px.bar(monthly_returns, x=monthly_returns.index.astype(str), y=monthly_returns.values, labels={'x': 'Month', 'y': 'Monthly Return'})
    st.plotly_chart(fig)

    # Distribution of Monthly Returns
    st.subheader("Distribution of Monthly Returns")
    monthly_returns = df.groupby('Month')['Daily Return'].sum()
    fig = px.histogram(monthly_returns, x=monthly_returns, nbins=20, labels={'x': 'Monthly Return'})
    st.plotly_chart(fig)


    # Drawdown Period starting August 2024
    st.subheader("Drawdown (Starting August 2024)")
    df_aug = df[df['Entry time'] >= '2024-08-01']
    drawdown = calculate_drawdown(df_aug['Cumulative Return'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_aug['Entry time'], y=drawdown, mode='lines', name='Drawdown'))
    st.plotly_chart(fig)
    
    # Leverage Usage (if leverage data exists)
    if 'Leverage' in df.columns:
        st.subheader("Leverage Usage")
        fig = px.line(df, x='Entry time', y='Leverage', title="Leverage Usage Over Time")
        st.plotly_chart(fig)

    # Calculate Sharpe Ratio for each ticker based on monthly returns
    df_monthly_returns = df.groupby(['Ticker', 'Month'])['PNL'].sum().reset_index()
    ticker_sharpe_monthly = df_monthly_returns.groupby('Ticker')['PNL'].apply(lambda x: x.mean() / x.std())

    # Display Sharpe Ratio for each Ticker
    st.subheader("Sharpe Ratio for Each Ticker (Based on Monthly Returns)")
    st.write(ticker_sharpe_monthly)


