import requests
import pandas as pd
from datetime import date

def fetch_fed_funds_rate():
    # FRED API endpoint
    url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Parameters for the API request
    params = {
        "series_id": "DFF",  # Series ID for Daily Federal Funds Rate
        "api_key": "e3434929fa58147c9ba379384c6cc9f1",  # Replace with your FRED API key
        "file_type": "json",
        "observation_start": date(date.today().year - 50, date.today().month, date.today().day).isoformat(),
        "observation_end": date.today().isoformat()
    }

    # Make the API request
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['observations'])
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'])
        
        # Sort by date and get the most recent non-null value
        df = df.sort_values('date', ascending=False)
        latest_rate = df[df['value'].notnull()]['value'].iloc[0]
        
        return latest_rate, df
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

if __name__ == "__main__":
    rate = fetch_fed_funds_rate()
    if rate is not None:
        print(f"The latest Federal Funds Rate is: {rate}%")

_, df = fetch_fed_funds_rate()