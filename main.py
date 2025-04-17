from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import requests
import json
from dateutil.relativedelta import relativedelta
import io

app = FastAPI(
    title="Portfolio Optimizer API",
    description="API for portfolio optimization and analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EODHD_API_KEY = '66b4ad3791f886.55196687'

# Models
class Portfolio(BaseModel):
    tickers: List[str]
    weights: List[float]

class OptimizationRequest(Portfolio):
    start_date: datetime
    end_date: datetime

class PerformanceMetrics(BaseModel):
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float

# Helper functions from optimizer.py
def get_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data for given tickers using EODHD API."""
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    all_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            url = f"https://eodhd.com/api/eod/{ticker}?from={start_str}&to={end_str}&api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                ticker_data = response.json()
                df = pd.DataFrame(ticker_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df = df[['adjusted_close']]
                df.columns = [ticker]
                
                if all_data.empty:
                    all_data = df
                else:
                    all_data = all_data.join(df, how='outer')
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error fetching data for {ticker}: {response.status_code} - {response.text}"
                )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Exception fetching data for {ticker}: {str(e)}"
            )
    
    return all_data.fillna(method='ffill')

def calculate_returns(prices):
    """Calculate daily returns from price data."""
    return prices.pct_change().dropna()

def portfolio_performance(weights, returns):
    """Calculate portfolio performance metrics."""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_volatility
    return portfolio_return, portfolio_volatility, sharpe_ratio

def optimize_portfolio(returns):
    """Perform mean-variance optimization to maximize Sharpe ratio."""
    num_assets = len(returns.columns)
    args = (returns,)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    def negative_sharpe(weights, returns):
        return -portfolio_performance(weights, returns)[2]
    
    initial_guess = np.array([1/num_assets] * num_assets)
    result = minimize(negative_sharpe, initial_guess, args=args, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    return result['x']

async def verify_tickers(tickers):
    """Verify if tickers are valid and can be found on EODHD."""
    valid_tickers = []
    invalid_tickers = []
    
    for ticker in tickers:
        try:
            url = f"https://eodhd.com/api/real-time/{ticker}?api_token={EODHD_API_KEY}&fmt=json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'code' in data and data.get('code') != 'error':
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            else:
                invalid_tickers.append(ticker)
        except Exception:
            invalid_tickers.append(ticker)
    
    return valid_tickers, invalid_tickers

# API Endpoints
@app.post("/api/portfolio/verify")
async def verify_portfolio(portfolio: Portfolio):
    """Verify if the provided tickers are valid"""
    valid_tickers, invalid_tickers = await verify_tickers(portfolio.tickers)
    return {
        "valid_tickers": valid_tickers,
        "invalid_tickers": invalid_tickers,
        "total_percentage": sum(portfolio.weights) * 100
    }

@app.post("/api/portfolio/optimize")
async def optimize_portfolio_endpoint(request: OptimizationRequest):
    """Optimize portfolio and return performance metrics"""
    try:
        # Get historical data
        prices = get_stock_data(request.tickers, request.start_date, request.end_date)
        returns = calculate_returns(prices)
        
        # Calculate original portfolio metrics
        orig_return, orig_vol, orig_sharpe = portfolio_performance(
            np.array(request.weights), returns
        )
        
        # Optimize portfolio
        opt_weights = optimize_portfolio(returns)
        opt_return, opt_vol, opt_sharpe = portfolio_performance(opt_weights, returns)
        
        return {
            "original_portfolio": {
                "metrics": PerformanceMetrics(
                    expected_annual_return=orig_return,
                    annual_volatility=orig_vol,
                    sharpe_ratio=orig_sharpe
                ),
                "weights": dict(zip(request.tickers, request.weights))
            },
            "optimized_portfolio": {
                "metrics": PerformanceMetrics(
                    expected_annual_return=opt_return,
                    annual_volatility=opt_vol,
                    sharpe_ratio=opt_sharpe
                ),
                "weights": dict(zip(request.tickers, opt_weights.tolist()))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/efficient-frontier")
async def get_efficient_frontier(request: OptimizationRequest):
    """Generate efficient frontier points"""
    try:
        prices = get_stock_data(request.tickers, request.start_date, request.end_date)
        returns = calculate_returns(prices)
        
        num_portfolios = 500
        results = []
        
        for _ in range(num_portfolios):
            weights = np.random.random(len(returns.columns))
            weights /= np.sum(weights)
            
            portfolio_return, portfolio_volatility, sharpe_ratio = portfolio_performance(
                weights, returns
            )
            
            results.append({
                "return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "weights": dict(zip(request.tickers, weights.tolist()))
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/historical-performance")
async def get_historical_performance(request: OptimizationRequest):
    """Get historical performance comparison"""
    try:
        prices = get_stock_data(request.tickers, request.start_date, request.end_date)
        returns = calculate_returns(prices)
        
        # Original portfolio performance
        portfolio_values = (prices * np.array(request.weights)).sum(axis=1)
        portfolio_returns = portfolio_values.pct_change().dropna()
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        # Optimized portfolio performance
        opt_weights = optimize_portfolio(returns)
        opt_values = (prices * opt_weights).sum(axis=1)
        opt_returns = opt_values.pct_change().dropna()
        opt_cumulative_returns = (1 + opt_returns).cumprod() - 1
        
        return {
            "dates": portfolio_values.index.strftime('%Y-%m-%d').tolist(),
            "original_portfolio": {
                "values": portfolio_values.tolist(),
                "returns": portfolio_returns.tolist(),
                "cumulative_returns": cumulative_returns.tolist()
            },
            "optimized_portfolio": {
                "values": opt_values.tolist(),
                "returns": opt_returns.tolist(),
                "cumulative_returns": opt_cumulative_returns.tolist()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/portfolio/upload")
async def upload_portfolio(file: UploadFile = File(...)):
    """Upload portfolio CSV file"""
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        if not all(col in df.columns for col in ['Ticker', 'Weight (%)']):
            raise HTTPException(
                status_code=400,
                detail="CSV must contain 'Ticker' and 'Weight (%)' columns"
            )
        
        return {
            "tickers": df['Ticker'].tolist(),
            "weights": (df['Weight (%)'].values / 100).tolist(),
            "total_percentage": df['Weight (%)'].sum()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/example-csv")
async def get_example_csv():
    """Get example portfolio CSV"""
    example_df = pd.DataFrame({
        'Ticker': ['AAPL.US', 'MSFT.US', 'GOOGL.US', 'AMZN.US', 'META.US'],
        'Weight (%)': [20, 25, 15, 25, 15]
    })
    
    return {
        "csv_content": example_df.to_csv(index=False),
        "filename": "example_portfolio.csv"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)