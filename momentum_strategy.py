import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Momentum Factor Backtesting Tool", layout="wide")

# API Key - in production, this should be stored more securely
API_KEY = "666bf36deeedc4.45950582"

# Function to fetch data from EODHD
def fetch_eod_data(ticker, start_date, end_date):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    url = f'https://eodhd.com/api/eod/{ticker}?api_token={API_KEY}&fmt=json&from={start_str}&to={end_str}'
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        else:
            st.error(f"Error fetching data: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Calculate momentum
def calculate_momentum(df, lookback_period):
    """
    Calculate momentum factor for the given lookback period
    Normalized by volatility
    """
    # Calculate returns
    df['return'] = df['adjusted_close'].pct_change()
    
    # Calculate momentum (using price ratio instead of return)
    df[f'momentum_{lookback_period}m'] = df['adjusted_close'] / df['adjusted_close'].shift(lookback_period)
    
    # Calculate volatility (standard deviation of returns over the same period)
    df[f'volatility_{lookback_period}m'] = df['return'].rolling(window=lookback_period).std()
    
    # Normalize momentum by volatility
    df[f'momentum_{lookback_period}m_norm'] = df[f'momentum_{lookback_period}m'] / df[f'volatility_{lookback_period}m']
    
    return df

# Backtest strategy
def backtest_momentum_strategy(df, lookback_period, rebalance_period=1, top_pct=0.2):
    """
    Backtest a momentum strategy
    lookback_period: Period in months to calculate momentum
    rebalance_period: How often to rebalance the portfolio (in months)
    top_pct: Percentage of top momentum stocks to include in the portfolio
    """
    # Make sure we have momentum data calculated
    if f'momentum_{lookback_period}m_norm' not in df.columns:
        df = calculate_momentum(df, lookback_period)
    
    # Create a copy of the dataframe with only month-end dates
    monthly_df = df.resample('M').last()
    
    # Initialize portfolio
    portfolio_value = 100.0  # Starting with $100
    portfolio_values = []
    dates = []
    
    # Loop through each rebalance date
    for i in range(rebalance_period, len(monthly_df), rebalance_period):
        # Get the current month data
        current_date = monthly_df.index[i]
        
        # Get previous momentum values
        momentum_value = monthly_df.iloc[i][f'momentum_{lookback_period}m_norm']
        
        # Calculate return for the next period
        if i + rebalance_period < len(monthly_df):
            future_return = monthly_df.iloc[i+rebalance_period]['adjusted_close'] / monthly_df.iloc[i]['adjusted_close'] - 1
            portfolio_value *= (1 + future_return)
        
        portfolio_values.append(portfolio_value)
        dates.append(current_date)
    
    # Create a dataframe with portfolio values
    portfolio_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values
    })
    portfolio_df.set_index('date', inplace=True)
    
    # Calculate benchmark (buy and hold)
    benchmark_start = monthly_df.iloc[rebalance_period]['adjusted_close']
    benchmark_end = monthly_df.iloc[-1]['adjusted_close']
    benchmark_return = benchmark_end / benchmark_start
    
    # Calculate performance metrics
    total_return = portfolio_value / 100.0 - 1
    num_years = len(portfolio_values) * rebalance_period / 12
    annualized_return = (1 + total_return) ** (1 / num_years) - 1
    
    # Calculate volatility
    portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
    volatility = portfolio_returns.std() * np.sqrt(12 / rebalance_period)
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    return portfolio_df, total_return, annualized_return, volatility, sharpe_ratio, benchmark_return

# Main app
def main():
    st.title("Momentum Factor Backtesting Tool")
    
    st.sidebar.header("Parameters")
    
    # Input parameters
    ticker = st.sidebar.text_input("Ticker Symbol", "SPY")
    
    # Date range
    today = datetime.date.today()
    years_ago = today - relativedelta(years=10)
    
    start_date = st.sidebar.date_input("Start Date", years_ago)
    end_date = st.sidebar.date_input("End Date", today)
    
    # Convert to datetime
    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    
    # Strategy parameters
    momentum_option = st.sidebar.selectbox("Momentum Period", ["6 Month", "12 Month"])
    lookback_period = 6 if momentum_option == "6 Month" else 12
    
    rebalance_period = st.sidebar.slider("Rebalance Period (Months)", 1, 12, 1)
    
    # Fetch data button
    if st.sidebar.button("Run Backtest"):
        # Adjust start date to include lookback period
        adjusted_start = start_date - relativedelta(months=lookback_period)
        
        # Display the loading message
        with st.spinner(f"Fetching data for {ticker}..."):
            # Fetch data
            df = fetch_eod_data(ticker, adjusted_start, end_date)
            
            if df is not None and not df.empty:
                # Calculate momentum
                df = calculate_momentum(df, lookback_period)
                
                # Display raw data
                with st.expander("Raw Data"):
                    st.dataframe(df)
                
                # Run backtest
                portfolio_df, total_return, annualized_return, volatility, sharpe_ratio, benchmark_return = backtest_momentum_strategy(
                    df, lookback_period, rebalance_period)
                
                # Display results
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Return", f"{total_return:.2%}")
                col2.metric("Annualized Return", f"{annualized_return:.2%}")
                col3.metric("Volatility", f"{volatility:.2%}")
                col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
                # Compare to benchmark
                st.subheader("Comparison to Buy & Hold")
                col1, col2 = st.columns(2)
                col1.metric("Strategy Return", f"{total_return:.2%}")
                col2.metric("Benchmark Return", f"{benchmark_return - 1:.2%}")
                
                # Plot the results
                fig = make_subplots(rows=2, cols=1, 
                                   shared_xaxes=True,
                                   vertical_spacing=0.1,
                                   subplot_titles=('Portfolio Value', f'{momentum_option} Momentum Factor'))
                
                # Portfolio value
                fig.add_trace(
                    go.Scatter(x=portfolio_df.index, y=portfolio_df['portfolio_value'], name="Portfolio Value"),
                    row=1, col=1
                )
                
                # Momentum indicator
                momentum_df = df.resample('M').last()
                fig.add_trace(
                    go.Scatter(x=momentum_df.index, y=momentum_df[f'momentum_{lookback_period}m_norm'], name=f"{momentum_option} Momentum"),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, width=800)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the final portfolio dataframe
                with st.expander("Portfolio Performance Data"):
                    st.dataframe(portfolio_df)
            else:
                st.error("No data available for the selected parameters.")

if __name__ == "__main__":
    main()