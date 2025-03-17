import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd

def get_stock_data_eodhd(ticker_list, start_date, end_date, api_key):
    """Fetch historical price data for tickers from EOD Historical Data."""
    data = {}
    
    # Convert dates to string format
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    for ticker in ticker_list:
        try:
            url = f'https://eodhd.com/api/eod/{ticker}?api_token={api_key}&fmt=json&from={start_str}&to={end_str}'
            response = requests.get(url)
            
            if response.status_code == 200:
                ticker_data = pd.DataFrame(response.json())
                
                # Check if data exists
                if not ticker_data.empty:
                    # Convert date string to datetime
                    ticker_data['date'] = pd.to_datetime(ticker_data['date'])
                    ticker_data = ticker_data.set_index('date')
                    
                    # Use adjusted close price
                    data[ticker] = ticker_data['adjusted_close']
                else:
                    st.warning(f"No data found for {ticker}")
            else:
                st.error(f"Error fetching {ticker}: HTTP Status {response.status_code}")
                
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    
    if data:
        # Create a DataFrame with all tickers and align on dates
        prices_df = pd.DataFrame(data)
        return prices_df
    else:
        return None

def calculate_momentum(prices_df, lookback_period):
    """Calculate momentum returns for a given lookback period."""
    # Convert lookback period from months to days approximately
    lookback_days = lookback_period * 30
    
    # Use rolling window to find the price lookback_days ago
    momentum = prices_df.pct_change(lookback_days).dropna()
    return momentum

def create_momentum_portfolio(prices_df, lookback_period, rebalance_freq, top_n_pct, start_date, end_date):
    """Create a portfolio based on momentum strategy."""
    # Calculate momentum at each rebalance date
    current_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Need enough data for lookback
    first_possible_date = prices_df.index[0] + pd.DateOffset(months=lookback_period)
    if current_date < first_possible_date:
        current_date = first_possible_date
    
    portfolio_returns = []
    rebalance_dates = []
    holdings_history = {}
    
    while current_date <= end_date:
        # Find the closest actual trading day
        closest_date = prices_df.index[prices_df.index <= current_date][-1] if any(prices_df.index <= current_date) else None
        
        if closest_date is None:
            current_date += pd.DateOffset(months=rebalance_freq)
            continue
            
        # Calculate momentum up to this date
        historical_data = prices_df.loc[:closest_date]
        momentum_returns = calculate_momentum(historical_data, lookback_period)
        
        if not momentum_returns.empty and len(momentum_returns.index) > 0:
            # Get latest momentum values
            latest_momentum = momentum_returns.iloc[-1].dropna()
            
            # Skip if no valid momentum values
            if len(latest_momentum) == 0:
                current_date += pd.DateOffset(months=rebalance_freq)
                continue
                
            # Select top N% stocks based on momentum
            top_n = max(1, int(len(latest_momentum) * (top_n_pct / 100)))
            top_stocks = latest_momentum.nlargest(top_n).index.tolist()
            
            # Record holdings for this rebalance date
            holdings_history[closest_date] = top_stocks
            rebalance_dates.append(closest_date)
        
        # Move to next rebalance date
        current_date += pd.DateOffset(months=rebalance_freq)
    
    # Calculate portfolio performance
    portfolio_values = []
    current_value = 100  # Start with $100
    benchmark_values = []
    benchmark_value = 100  # Start with $100
    
    dates = []
    
    for i in range(len(rebalance_dates)):
        current_date = rebalance_dates[i]
        
        # Determine end date for this period
        if i < len(rebalance_dates) - 1:
            end_date = rebalance_dates[i + 1]
        else:
            end_date = prices_df.index[-1]
        
        # Get returns for this period
        period_prices = prices_df.loc[current_date:end_date]
        
        # Skip if we don't have enough data
        if len(period_prices) <= 1:
            continue
        
        # Calculate portfolio returns for this period
        portfolio_stocks = holdings_history[current_date]
        
        # Filter stocks that exist in the data
        valid_stocks = [stock for stock in portfolio_stocks if stock in period_prices.columns]
        
        if not valid_stocks:
            continue
            
        # Calculate daily returns
        for date in period_prices.index[1:]:  # Skip the first day
            # Ensure we have data for these stocks on this date
            available_stocks = [stock for stock in valid_stocks if stock in period_prices.columns and not pd.isna(period_prices.loc[date, stock])]
            
            if not available_stocks:
                continue
                
            stock_returns = period_prices.loc[date, available_stocks].pct_change().mean()
            current_value *= (1 + stock_returns)
            
            # Calculate benchmark (equal-weighted all stocks)
            available_bench_stocks = [col for col in period_prices.columns if not pd.isna(period_prices.loc[date, col])]
            if available_bench_stocks:
                bench_return = period_prices.loc[date, available_bench_stocks].pct_change().mean()
                benchmark_value *= (1 + bench_return)
            
            portfolio_values.append(current_value)
            benchmark_values.append(benchmark_value)
            dates.append(date)
    
    # Create performance dataframe
    if dates:
        performance_df = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values,
            'Benchmark Value': benchmark_values
        }).set_index('Date')
        return performance_df, holdings_history
    else:
        return pd.DataFrame(columns=['Portfolio Value', 'Benchmark Value']), holdings_history

def calculate_performance_metrics(performance_df):
    """Calculate key performance metrics for the portfolio."""
    if performance_df.empty or len(performance_df) < 2:
        # Return empty metrics if not enough data
        empty_metrics = pd.DataFrame({
            'Total Return': [0, 0],
            'Annualized Return (%)': [0, 0],
            'Annualized Volatility (%)': [0, 0],
            'Sharpe Ratio': [0, 0],
            'Max Drawdown (%)': [0, 0]
        }, index=['Portfolio', 'Benchmark'])
        return empty_metrics
    
    portfolio_returns = performance_df['Portfolio Value'].pct_change().dropna()
    benchmark_returns = performance_df['Benchmark Value'].pct_change().dropna()
    
    # Calculate metrics
    total_return_port = (performance_df['Portfolio Value'].iloc[-1] / performance_df['Portfolio Value'].iloc[0]) - 1
    total_return_bench = (performance_df['Benchmark Value'].iloc[-1] / performance_df['Benchmark Value'].iloc[0]) - 1
    
    # Annualized returns
    days = (performance_df.index[-1] - performance_df.index[0]).days
    ann_return_port = (1 + total_return_port) ** (365 / max(days, 1)) - 1
    ann_return_bench = (1 + total_return_bench) ** (365 / max(days, 1)) - 1
    
    # Volatility
    ann_vol_port = portfolio_returns.std() * np.sqrt(252) if len(portfolio_returns) > 0 else 0
    ann_vol_bench = benchmark_returns.std() * np.sqrt(252) if len(benchmark_returns) > 0 else 0
    
    # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_port = ann_return_port / ann_vol_port if ann_vol_port > 0 else 0
    sharpe_bench = ann_return_bench / ann_vol_bench if ann_vol_bench > 0 else 0
    
    # Maximum drawdown
    if len(portfolio_returns) > 0:
        cum_returns_port = (1 + portfolio_returns).cumprod()
        running_max_port = cum_returns_port.cummax()
        drawdown_port = (cum_returns_port / running_max_port) - 1
        max_drawdown_port = drawdown_port.min()
    else:
        max_drawdown_port = 0
        
    if len(benchmark_returns) > 0:
        cum_returns_bench = (1 + benchmark_returns).cumprod()
        running_max_bench = cum_returns_bench.cummax()
        drawdown_bench = (cum_returns_bench / running_max_bench) - 1
        max_drawdown_bench = drawdown_bench.min()
    else:
        max_drawdown_bench = 0
    
    metrics = {
        'Total Return': [total_return_port * 100, total_return_bench * 100],
        'Annualized Return (%)': [ann_return_port * 100, ann_return_bench * 100],
        'Annualized Volatility (%)': [ann_vol_port * 100, ann_vol_bench * 100],
        'Sharpe Ratio': [sharpe_port, sharpe_bench],
        'Max Drawdown (%)': [max_drawdown_port * 100, max_drawdown_bench * 100]
    }
    
    return pd.DataFrame(metrics, index=['Portfolio', 'Benchmark'])

def main():
    st.title("Momentum Strategy Backtester")
    
    st.sidebar.header("Strategy Parameters")
    
    # API Key
    api_key = st.sidebar.text_input("EODHD API Key", value="666bf36deeedc4.45950582", type="password")
    
    # Input for tickers
    ticker_input = st.sidebar.text_area("Enter ticker symbols (comma-separated):", 
                                       value="AAPL.US,MSFT.US,AMZN.US,GOOGL.US,META.US,TSLA.US,NVDA.US,JNJ.US,V.US,PG.US")
    
    tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]
    
    # Date range
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start Date", datetime(2018, 1, 1))
    end_date = col2.date_input("End Date", datetime.now())
    
    # Lookback period
    lookback_period = st.sidebar.radio("Momentum Lookback Period", [6, 12], 
                                      format_func=lambda x: f"{x} Months")
    
    # Rebalance frequency
    rebalance_freq = st.sidebar.slider("Rebalance Frequency (Months)", 1, 12, 3)
    
    # Portfolio size
    top_n_pct = st.sidebar.slider("Top Momentum Stocks (%)", 10, 50, 30)
    
    # Backtest button
    if st.sidebar.button("Run Backtest"):
        with st.spinner('Fetching data and running backtest...'):
            # Extend start date to include lookback period
            adjusted_start_date = start_date - timedelta(days=lookback_period * 31)
            
            # Get stock data
            prices_df = get_stock_data_eodhd(tickers, adjusted_start_date, end_date, api_key)
            
            if prices_df is not None and not prices_df.empty:
                st.success(f"Data fetched for {len(prices_df.columns)} stocks")
                
                # Run backtest
                performance_df, holdings_history = create_momentum_portfolio(
                    prices_df, 
                    lookback_period, 
                    rebalance_freq, 
                    top_n_pct, 
                    start_date, 
                    end_date
                )
                
                if not performance_df.empty:
                    # Display results
                    st.header("Backtest Results")
                    
                    # Performance chart
                    st.subheader("Portfolio Performance")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=performance_df.index, 
                                            y=performance_df['Portfolio Value'], 
                                            name="Momentum Portfolio"))
                    fig.add_trace(go.Scatter(x=performance_df.index, 
                                            y=performance_df['Benchmark Value'], 
                                            name="Equal-Weight Benchmark"))
                    fig.update_layout(title="Portfolio vs Benchmark Performance",
                                    xaxis_title="Date",
                                    yaxis_title="Value ($)",
                                    legend_title="Portfolio",
                                    height=500)
                    st.plotly_chart(fig)
                    
                    # Performance metrics
                    st.subheader("Performance Metrics")
                    metrics_df = calculate_performance_metrics(performance_df)
                    st.dataframe(metrics_df.style.format({
                        'Total Return': '{:.2f}%',
                        'Annualized Return (%)': '{:.2f}%',
                        'Annualized Volatility (%)': '{:.2f}%',
                        'Sharpe Ratio': '{:.2f}',
                        'Max Drawdown (%)': '{:.2f}%'
                    }))
                    
                    # Holdings history
                    if holdings_history:
                        st.subheader("Portfolio Holdings History")
                        holdings_df = pd.DataFrame(index=prices_df.columns)
                        
                        for date, stocks in holdings_history.items():
                            holdings_df[date.strftime('%Y-%m-%d')] = holdings_df.index.isin(stocks)
                        
                        # Convert boolean to markers for better visualization
                        holdings_df = holdings_df.replace({True: '✓', False: ''})
                        
                        st.dataframe(holdings_df)
                else:
                    st.warning("Not enough data to calculate performance. Try extending the date range or checking stock symbols.")
            else:
                st.error("Failed to fetch data. Please check your API key and ticker symbols.")
    
    # Add instructions
    with st.expander("Instructions"):
        st.markdown("""
        ### How to use this tool:
        
        1. **Enter your EODHD API Key** if different from the default.
        2. **Enter ticker symbols** in the sidebar (comma-separated). These should be valid EODHD tickers with exchange suffix (e.g., AAPL.US, VOD.L).
        3. **Choose momentum lookback period** - 6 or 12 months.
        4. **Set rebalance frequency** - how often to update the portfolio (in months).
        5. **Select top momentum percentage** - what percentage of stocks with the highest momentum to include in the portfolio.
        6. **Click "Run Backtest"** to see the results.
        
        ### Interpretation:
        - The chart shows the performance of your momentum portfolio vs. an equal-weighted benchmark.
        - The metrics table provides key performance indicators.
        - The holdings history shows which stocks were in the portfolio at each rebalance date.
        """)

if __name__ == "__main__":
    main()