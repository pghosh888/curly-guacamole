import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from datetime import datetime, timedelta
import io
import plotly.graph_objects as go
import plotly.express as px
from dateutil.relativedelta import relativedelta
import requests
import json

# Set page configuration
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration for EODHD API
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False


# Define API key and base URL
EODHD_API_KEY = '66b4ad3791f886.55196687'
#EODHD_API_KEY = st.secrets.get("EODHD_API_KEY", "YOUR_API_KEY")

# Define functions for portfolio optimization
def get_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data for given tickers using EODHD API."""
    # Format dates for EODHD API
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Create an empty DataFrame to store results
    all_data = pd.DataFrame()
    
    # Fetch data for each ticker
    for ticker in tickers:
        try:
            # Make API request
            url = f"https://eodhd.com/api/eod/{ticker}?from={start_str}&to={end_str}&api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                # Parse JSON response
                ticker_data = response.json()
                
                # Convert to DataFrame
                df = pd.DataFrame(ticker_data)
                
                # Convert date string to datetime
                df['date'] = pd.to_datetime(df['date'])
                
                # Set date as index
                df.set_index('date', inplace=True)
                
                # Keep only adjusted close price
                df = df[['adjusted_close']]
                
                # Rename column to ticker name
                df.columns = [ticker]
                
                # Join with main DataFrame
                if all_data.empty:
                    all_data = df
                else:
                    all_data = all_data.join(df, how='outer')
            else:
                st.warning(f"Error fetching data for {ticker}: {response.status_code} - {response.text}")
                
        except Exception as e:
            st.warning(f"Exception fetching data for {ticker}: {e}")
    
    # Forward fill any missing values (weekends, holidays)
    all_data = all_data.fillna(method='ffill')
    
    return all_data

def calculate_returns(prices):
    """Calculate daily returns from price data."""
    returns = prices.pct_change().dropna()
    return returns

def portfolio_performance(weights, returns):
    """Calculate portfolio performance metrics."""
    # Calculate portfolio return and volatility
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = portfolio_return / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio

def negative_sharpe(weights, returns):
    """Objective function for optimization: negative Sharpe ratio."""
    return -portfolio_performance(weights, returns)[2]

def optimize_portfolio(returns):
    """Perform mean-variance optimization to maximize Sharpe ratio."""
    num_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    
    # Initial guess: equal weights
    initial_guess = np.array([1/num_assets] * num_assets)
    
    # Optimize
    result = minimize(negative_sharpe, initial_guess, args=args, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    return result['x']

def historical_portfolio_performance(weights, tickers, start_date, end_date):
    """Calculate historical performance of portfolio with given weights."""
    # Get historical data
    prices = get_stock_data(tickers, start_date, end_date)
    
    # Create portfolio performance series
    portfolio_values = (prices * weights).sum(axis=1)
    portfolio_returns = portfolio_values.pct_change().dropna()
    
    # Calculate cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    return portfolio_values, portfolio_returns, cumulative_returns

def verify_tickers(tickers):
    """Verify if tickers are valid and can be found on EODHD."""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            # Try to get the latest price to verify ticker exists
            url = f"https://eodhd.com/api/real-time/{ticker}?api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                # Check if we got valid data back (not an error message)
                if isinstance(data, dict) and 'code' in data and data.get('code') != 'error':
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception as e:
            st.warning(f"Error validating ticker {ticker}: {e}")
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

# Initialize session state for multi-page functionality
if 'page' not in st.session_state:
    st.session_state.page = 1

if 'portfolio_df' not in st.session_state:
    st.session_state.portfolio_df = None

if 'original_weights' not in st.session_state:
    st.session_state.original_weights = None

if 'optimized_weights' not in st.session_state:
    st.session_state.optimized_weights = None

if 'tickers' not in st.session_state:
    st.session_state.tickers = None

if 'valid_tickers' not in st.session_state:
    st.session_state.valid_tickers = None

if 'invalid_tickers' not in st.session_state:
    st.session_state.invalid_tickers = None

if 'total_percentage' not in st.session_state:
    st.session_state.total_percentage = 0

# Navigation functions
def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

def go_to_page(page_num):
    st.session_state.page = page_num

# Sidebar navigation menu
st.sidebar.title("Navigation")
st.sidebar.write("Current Page: ", st.session_state.page)

if st.sidebar.button("1. Upload Portfolio"):
    go_to_page(1)

if st.sidebar.button("2. Verify Portfolio"):
    go_to_page(2)

if st.sidebar.button("3. Optimize Portfolio"):
    go_to_page(3)

if st.sidebar.button("4. Performance Report"):
    go_to_page(4)

# Page 1: Upload Portfolio
if st.session_state.page == 1:
    st.title("Upload Your Portfolio")
    
    # Check if API key is needed
    #if not st.secrets.get("EODHD_API_KEY", "") and not st.session_state.api_key_set:
    st.info("This application requires an EODHD API key to fetch stock data.")
    api_key = st.text_input("Enter your EODHD API key:", 
                            value='66b4ad3791f886.55196687',
                            type="password")
        
    if st.button("Set API Key"):
        if api_key:
            #global EODHD_API_KEY
            EODHD_API_KEY = api_key
            st.session_state.api_key_input = api_key
            st.session_state.api_key_set = True
            st.success("API key set successfully!")
            st.rerun()
        else:
            st.error("Please enter a valid API key")
    
    if api_key:
        st.write("""
        Please upload a CSV file containing your stock portfolio. 
        The CSV should include columns for 'Ticker' and 'Weight (%)'.
        For EODHD, make sure to use the correct ticker format (e.g., AAPL.US, MSFT.US).
        """)
    
    # Example CSV download
    example_df = pd.DataFrame({
        'Ticker': ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'META.US'],
        'Weight (%)': [20, 25, 15, 25, 15]
    })
    
    csv_example = example_df.to_csv(index=False)
    st.download_button(
        label="Download Example CSV",
        data=csv_example,
        file_name="example_portfolio.csv",
        mime="text/csv"
    )
    
    # CSV upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = ['Ticker', 'Weight (%)']
            
            # Check if required columns exist
            if all(col in df.columns for col in required_columns):
                # Display the uploaded data
                st.write("Your uploaded portfolio:")
                st.dataframe(df)
                
                # Save to session state
                st.session_state.portfolio_df = df.copy()
                st.session_state.tickers = df['Ticker'].tolist()
                st.session_state.original_weights = df['Weight (%)'].values / 100
                st.session_state.total_percentage = df['Weight (%)'].sum()
                
                # Continue button
                if st.button("Continue to Verification"):
                    next_page()
            else:
                st.error(f"CSV must contain these columns: {', '.join(required_columns)}")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    
# Page 2: Verification Step
elif st.session_state.page == 2:
    st.title("Portfolio Verification")
    
    if st.session_state.portfolio_df is None:
        st.error("No portfolio data found. Please upload your portfolio first.")
        if st.button("Go to Upload Page"):
            go_to_page(1)
    else:
        # Calculate total percentage
        total_pct = st.session_state.total_percentage
        st.write(f"Total allocation: {total_pct}%")
        
        if abs(total_pct - 100) > 0.01:
            st.error(f"Portfolio allocation does not sum to 100%. Current sum: {total_pct}%")
        else:
            st.success("Portfolio allocation correctly sums to 100%.")
        
        # Verify tickers
        st.write("Verifying ticker symbols...")
        
        valid_tickers, invalid_tickers = verify_tickers(st.session_state.tickers)
        
        # Store results in session state
        st.session_state.valid_tickers = valid_tickers
        st.session_state.invalid_tickers = invalid_tickers
        
        if invalid_tickers:
            st.error(f"Invalid ticker symbols found: {', '.join(invalid_tickers)}")
            st.write("Please go back and correct these ticker symbols.")
            
            if st.button("Go Back to Upload"):
                go_to_page(1)
        else:
            st.success(f"All {len(valid_tickers)} ticker symbols are valid.")
            
            # Only show continue button if all tickers are valid and total is 100%
            if abs(total_pct - 100) <= 0.01:
                if st.button("Continue to Optimization"):
                    next_page()
            else:
                st.warning("Please fix the allocation percentages before continuing.")
                if st.button("Go Back to Update Allocations"):
                    go_to_page(1)

# Page 3: Mean-Variance Optimization
elif st.session_state.page == 3:
    st.title("Portfolio Optimization")
    
    if st.session_state.valid_tickers is None or not st.session_state.valid_tickers:
        st.error("No valid tickers found. Please verify your portfolio first.")
        if st.button("Go to Verification Page"):
            go_to_page(2)
    else:
        st.write("Performing mean-variance optimization to maximize Sharpe ratio...")
        
        try:
            # Get historical data for the past 2 years
            end_date = datetime.now()
            start_date = end_date - relativedelta(years=2)
            
            with st.spinner("Fetching historical stock data..."):
                prices = get_stock_data(st.session_state.valid_tickers, start_date, end_date)
                returns = calculate_returns(prices)
            
            # Get original portfolio weights aligned with valid tickers
            original_weights = []
            for ticker in st.session_state.valid_tickers:
                idx = st.session_state.tickers.index(ticker)
                original_weights.append(st.session_state.original_weights[idx])
                
            # Normalize original weights to sum to 1
            original_weights = np.array(original_weights) / sum(original_weights)
            
            # Calculate original portfolio performance
            orig_return, orig_vol, orig_sharpe = portfolio_performance(original_weights, returns)
            
            st.write("Current Portfolio Performance:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Annual Return", f"{orig_return:.2%}")
            with col2:
                st.metric("Annual Volatility", f"{orig_vol:.2%}")
            with col3:
                st.metric("Sharpe Ratio", f"{orig_sharpe:.2f}")
            
            # Perform optimization
            with st.spinner("Optimizing portfolio..."):
                optimized_weights = optimize_portfolio(returns)
                opt_return, opt_vol, opt_sharpe = portfolio_performance(optimized_weights, returns)
            
            st.write("Optimized Portfolio Performance:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Expected Annual Return", f"{opt_return:.2%}", 
                          delta=f"{opt_return - orig_return:.2%}")
            with col2:
                st.metric("Annual Volatility", f"{opt_vol:.2%}", 
                          delta=f"{opt_vol - orig_vol:.2%}", delta_color="inverse")
            with col3:
                st.metric("Sharpe Ratio", f"{opt_sharpe:.2f}", 
                          delta=f"{opt_sharpe - orig_sharpe:.2f}")
            
            # Display weights comparison
            weights_comparison = pd.DataFrame({
                'Ticker': st.session_state.valid_tickers,
                'Current Weight (%)': [w * 100 for w in original_weights],
                'Optimized Weight (%)': [w * 100 for w in optimized_weights],
                'Difference (%)': [(ow - cw) * 100 for ow, cw in zip(optimized_weights, original_weights)]
            })
            
            st.write("Weights Comparison:")
            st.dataframe(weights_comparison.style.format({
                'Current Weight (%)': '{:.2f}%',
                'Optimized Weight (%)': '{:.2f}%',
                'Difference (%)': '{:.2f}%'
            }))
            
            # Plot efficient frontier
            st.write("Efficient Frontier Visualization:")
            
            with st.spinner("Generating efficient frontier..."):
                # Generate random portfolios for visualization
                num_portfolios = 5
                results = np.zeros((3, num_portfolios))
                weights_record = []
                
                for i in range(num_portfolios):
                    weights = np.random.random(len(returns.columns))
                    weights /= np.sum(weights)
                    weights_record.append(weights)
                    
                    portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(weights, returns)
                    
                    results[0, i] = portfolio_volatility
                    results[1, i] = portfolio_return
                    results[2, i] = sharpe_ratio
                
                # Create a DataFrame for plotting
                results_df = pd.DataFrame({
                    'Volatility': results[0,:],
                    'Return': results[1,:],
                    'Sharpe': results[2,:]
                })
                
                # Plot with Plotly
                fig = px.scatter(
                    results_df, x='Volatility', y='Return', color='Sharpe',
                    color_continuous_scale='viridis',
                    title='Portfolio Optimization: Efficient Frontier'
                )
                
                # Add current and optimized portfolio points
                fig.add_trace(
                    go.Scatter(
                        x=[orig_vol], y=[orig_return],
                        mode='markers',
                        marker=dict(size=15, color='red'),
                        name='Current Portfolio'
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[opt_vol], y=[opt_return],
                        mode='markers',
                        marker=dict(size=15, color='green'),
                        name='Optimized Portfolio'
                    )
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Annual Volatility',
                    yaxis_title='Annual Return',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Save optimized weights for next page
            st.session_state.optimized_weights = optimized_weights
            
            if st.button("Continue to Performance Report"):
                next_page()
                
        except Exception as e:
            st.error(f"An error occurred during optimization: {e}")
            st.write("Please try again or adjust your portfolio.")

# Page 4: Performance Report
elif st.session_state.page == 4:
    st.title("Portfolio Performance Report")
    
    if st.session_state.optimized_weights is None:
        st.error("No optimization results found. Please optimize your portfolio first.")
        if st.button("Go to Optimization Page"):
            go_to_page(3)
    else:
        try:
            # Get 1 year historical data for performance comparison
            end_date = datetime.now()
            start_date = end_date - relativedelta(years=1)
            
            # Get original portfolio weights aligned with valid tickers
            original_weights = []
            for ticker in st.session_state.valid_tickers:
                idx = st.session_state.tickers.index(ticker)
                original_weights.append(st.session_state.original_weights[idx])
                
            # Normalize original weights to sum to 1
            original_weights = np.array(original_weights) / sum(original_weights)
            
            with st.spinner("Calculating historical performance..."):
                # Calculate historical performance for original portfolio
                orig_values, orig_returns, orig_cum_returns = historical_portfolio_performance(
                    original_weights, st.session_state.valid_tickers, start_date, end_date
                )
                
                # Calculate historical performance for optimized portfolio
                opt_values, opt_returns, opt_cum_returns = historical_portfolio_performance(
                    st.session_state.optimized_weights, st.session_state.valid_tickers, start_date, end_date
                )
            
            # Display performance comparison chart
            st.write("### One Year Performance Comparison")
            
            # Create DataFrame for plotting
            performance_df = pd.DataFrame({
                'Date': orig_cum_returns.index,
                'Original Portfolio': orig_cum_returns.values * 100,
                'Optimized Portfolio': opt_cum_returns.values * 100
            })
            
            fig = px.line(
                performance_df, x='Date', y=['Original Portfolio', 'Optimized Portfolio'],
                title='Cumulative Returns (%): Original vs. Optimized Portfolio',
                labels={'value': 'Cumulative Return (%)', 'variable': 'Portfolio Type'}
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate performance metrics
            metrics = pd.DataFrame({
                'Metric': ['Total Return (%)', 'Annualized Return (%)', 'Volatility (%)', 
                          'Sharpe Ratio', 'Max Drawdown (%)'],
                'Original Portfolio': [
                    orig_cum_returns.iloc[-1] * 100,
                    ((1 + orig_cum_returns.iloc[-1]) ** (252 / len(orig_cum_returns)) - 1) * 100,
                    orig_returns.std() * np.sqrt(252) * 100,
                    ((((1 + orig_cum_returns.iloc[-1]) ** (252 / len(orig_cum_returns)) - 1)) / 
                     (orig_returns.std() * np.sqrt(252))),
                    (orig_values / orig_values.cummax() - 1).min() * 100
                ],
                'Optimized Portfolio': [
                    opt_cum_returns.iloc[-1] * 100,
                    ((1 + opt_cum_returns.iloc[-1]) ** (252 / len(opt_cum_returns)) - 1) * 100,
                    opt_returns.std() * np.sqrt(252) * 100,
                    ((((1 + opt_cum_returns.iloc[-1]) ** (252 / len(opt_cum_returns)) - 1)) / 
                     (opt_returns.std() * np.sqrt(252))),
                    (opt_values / opt_values.cummax() - 1).min() * 100
                ],
                'Difference': [
                    (opt_cum_returns.iloc[-1] - orig_cum_returns.iloc[-1]) * 100,
                    (((1 + opt_cum_returns.iloc[-1]) ** (252 / len(opt_cum_returns)) - 1) - 
                     ((1 + orig_cum_returns.iloc[-1]) ** (252 / len(orig_cum_returns)) - 1)) * 100,
                    (opt_returns.std() - orig_returns.std()) * np.sqrt(252) * 100,
                    (((((1 + opt_cum_returns.iloc[-1]) ** (252 / len(opt_cum_returns)) - 1)) / 
                      (opt_returns.std() * np.sqrt(252))) - 
                     ((((1 + orig_cum_returns.iloc[-1]) ** (252 / len(orig_cum_returns)) - 1)) / 
                      (orig_returns.std() * np.sqrt(252)))),
                    ((opt_values / opt_values.cummax() - 1).min() - 
                     (orig_values / orig_values.cummax() - 1).min()) * 100
                ]
            })
            
            # Format the metrics
            metrics_styled = metrics.copy()
            for col in metrics.columns[1:]:
                metrics_styled[col] = metrics_styled[col].apply(lambda x: f"{x:.2f}")
            
            st.write("### Performance Metrics")
            st.table(metrics_styled.set_index('Metric'))
            
            # Display monthly returns heatmap
            st.write("### Monthly Returns Analysis")
            
            # Create monthly returns DataFrames
            orig_monthly = orig_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            opt_monthly = opt_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            
            # Format for display
            orig_monthly_df = pd.DataFrame({
                'Year': orig_monthly.index.year,
                'Month': orig_monthly.index.month,
                'Return (%)': orig_monthly.values
            })
            
            opt_monthly_df = pd.DataFrame({
                'Year': opt_monthly.index.year,
                'Month': opt_monthly.index.month,
                'Return (%)': opt_monthly.values
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Original Portfolio Monthly Returns (%)")
                
                # Create pivot table
                pivot_orig = orig_monthly_df.pivot_table(
                    index='Year', columns='Month', values='Return (%)'
                )
                pivot_orig.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Plot heatmap using plotly
                fig_orig = px.imshow(
                    pivot_orig,
                    text_auto='.1f',
                    color_continuous_scale='RdYlGn',
                    labels=dict(x="Month", y="Year", color="Return (%)"),
                    title="Original Portfolio Monthly Returns (%)"
                )
                
                st.plotly_chart(fig_orig, use_container_width=True)
                
            with col2:
                st.write("Optimized Portfolio Monthly Returns (%)")
                
                # Create pivot table
                pivot_opt = opt_monthly_df.pivot_table(
                    index='Year', columns='Month', values='Return (%)'
                )
                pivot_opt.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Plot heatmap using plotly
                fig_opt = px.imshow(
                    pivot_opt,
                    text_auto='.1f',
                    color_continuous_scale='RdYlGn',
                    labels=dict(x="Month", y="Year", color="Return (%)"),
                    title="Optimized Portfolio Monthly Returns (%)"
                )
                
                st.plotly_chart(fig_opt, use_container_width=True)
            
            # Analysis and Explanation
            st.write("## Portfolio Analysis and Insights")
            
            # Calculate some values for report
            return_improvement = metrics['Difference'][0]
            risk_change = metrics['Difference'][2]
            sharpe_improvement = metrics['Difference'][3]
            
            st.write("""
            ### Key Findings
            
            The mean-variance optimization has effectively rebalanced your portfolio to achieve a more optimal risk-return profile. Here's what the analysis reveals:
            """)
            
            if return_improvement > 0:
                st.write(f"✅ **Improved Returns:** The optimized portfolio demonstrated a {abs(return_improvement):.2f}% higher total return over the past year compared to your original allocation.")
            else:
                st.write(f"📊 **Return Trade-off:** The optimization resulted in a {abs(return_improvement):.2f}% lower total return over the past year, but this was balanced by risk reduction.")
                
            if risk_change < 0:
                st.write(f"✅ **Reduced Volatility:** The optimization successfully reduced portfolio volatility by {abs(risk_change):.2f}%, providing a smoother equity curve and more consistent returns.")
            else:
                st.write(f"📈 **Increased Volatility:** The optimization increased volatility by {risk_change:.2f}%, but this was offset by higher potential returns.")
                
            if sharpe_improvement > 0:
                st.write(f"✅ **Better Risk-Adjusted Returns:** The Sharpe ratio improved by {sharpe_improvement:.2f}, indicating a more efficient portfolio that delivers better returns per unit of risk.")
            else:
                st.write(f"⚠️ **Risk-Adjusted Performance:** The Sharpe ratio decreased by {abs(sharpe_improvement):.2f}, suggesting a potential need to reevaluate the optimization parameters.")
            
            st.write("""
            ### Weight Reallocation Logic
            
            The optimization algorithm has redistributed your investment across assets to find the mathematically optimal allocation that maximizes the Sharpe ratio. Here's what happened behind the scenes:
            
            1. **Correlation Analysis:** Assets with lower correlation to each other received preference to improve diversification.
            
            2. **Return-to-Risk Evaluation:** The algorithm favored assets that historically provided higher returns relative to their volatility.
            
            3. **Diversification Benefit:** The mathematical model identified the precise weight allocation that captures maximum diversification benefit.
            """)
            
            st.write("""
            ### Implementation Considerations
            
            Before implementing the optimized portfolio allocation, consider these important factors:
            
            1. **Transaction Costs:** Rebalancing to the optimized weights will incur trading costs which are not factored into this analysis.
            
            2. **Tax Implications:** Selling positions may trigger capital gains taxes depending on your jurisdiction.
            
            3. **Future Expectations:** This optimization is based on historical data and assumes similar patterns will continue. Consider adjusting if you have strong views on changing market conditions.
            
            4. **Rebalancing Schedule:** Determine how frequently you'll want to re-run the optimization and rebalance your portfolio (quarterly, semi-annually, or annually).
            """)
            
            st.write("""
            ### Next Steps
            
            1. Consider implementing the optimized weights gradually to minimize market impact costs.
            
            2. Set up a regular schedule to rerun this optimization to ensure your portfolio remains efficient as market conditions change.
            
            3. Compare actual performance against the projected performance to validate the model's effectiveness for your specific investments.
            """)
            
            # Reset button
            if st.button("Start Over with New Portfolio"):
                # Reset all relevant session state variables
                st.session_state.page = 1
                st.session_state.portfolio_df = None
                st.session_state.original_weights = None
                st.session_state.optimized_weights = None
                st.session_state.tickers = None
                st.session_state.valid_tickers = None
                st.session_state.invalid_tickers = None
                st.session_state.total_percentage = 0
                
                st.rerun()
                
        except Exception as e:
            st.error(f"An error occurred while generating the performance report: {e}")
            st.write("Please try again or go back to the optimization step.")

# Display progress bar at the bottom
progress_percentage = st.session_state.page * 25
st.progress(progress_percentage)