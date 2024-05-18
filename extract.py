import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the list of stock tickers and the time period
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'NVDA', 'NFLX', 'INTC', 'AMD',
    'IBM', 'ORCL', 'CSCO', 'ADBE', 'PYPL', 'QCOM', 'TXN', 'AVGO', 'CRM', 'SAP'
]
start_date = '2019-05-17'
end_date = '2024-05-17'  # Use today's date

# Function to download and format data for a single stock
def download_and_format_stock_data(ticker, start_date, end_date, retries=3):
    for i in range(retries):
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            if not stock_data.empty:
                stock_data.reset_index(inplace=True)
                stock_data['Name'] = ticker
                stock_data = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Name']]
                stock_data.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'name']
                return stock_data
        except Exception as e:
            print(f"Attempt {i+1} failed for {ticker}: {e}")
    return pd.DataFrame()

# Download and format data for all tickers
all_stocks_data = pd.concat([download_and_format_stock_data(ticker, start_date, end_date) for ticker in tickers])

# Display the first few rows of the combined data
print(all_stocks_data.head())

# Save to CSV
file_path = 'formatted_stock_data2.csv'
all_stocks_data.to_csv(file_path, index=False)

print(f"Data saved to {file_path}")
