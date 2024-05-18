import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# If you're using Jupyter Notebook, run this line to display the plots inline
%matplotlib inline


data = pd.read_csv("formatted_stock_data2.csv")
# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'])
data.head(10)

# Convert 'date' column to datetime if it's not already in datetime format
data['date'] = pd.to_datetime(data['date'])

# Group data by daily intervals and calculate statistics
daily_summary = data.groupby(pd.Grouper(key='date', freq='D')).agg(
    stats1D=('open', lambda x: np.nan if x.empty else (np.mean(x), np.std(x), pd.Series(x).skew()))
)

# Remove rows with NaN values (if any)
daily_summary = daily_summary.dropna()
# Print the result
print(daily_summary)

# Extracting values from tuple into separate columns
daily_summary[['mean_open', 'std_open', 'skewness_open']] = pd.DataFrame(
    daily_summary['stats1D'].tolist(), index=daily_summary.index
)

# Dropping the original 'stats1D' column
daily_summary.drop(columns=['stats1D'], inplace=True)

# Printing the updated DataFrame
print(daily_summary)

# Get the row count
row_count = len(data)

# Print formatted output
print("approximate_row_count")
print("----------------------")
print(row_count)

# Print summary statistics
print(data.describe())

import pandas as pd
import numpy as np

# Assuming stock_data is already defined and contains the dataset

# Convert 'date' column to datetime if it's not already in datetime format
data['date'] = pd.to_datetime(data['date'])

# Set 'date' as the index
data.set_index('date', inplace=True)

# Group data by daily intervals and calculate statistics for the 'open' price
daily_summary = data['open'].resample('D').agg(
    stats1D=lambda x: (np.mean(x), np.std(x), x.skew()) if not x.empty else np.nan
)

# Remove rows with NaN values (if any)
daily_summary = daily_summary.dropna()

# Print the result
print(daily_summary)


# Convert the 'stats1D' column tuples into separate columns
daily_summary[['mean_open', 'std_open', 'skewness_open']] = pd.DataFrame(
    daily_summary['stats1D'].tolist(), index=daily_summary.index
)

# Drop the original 'stats1D' column
daily_summary.drop(columns=['stats1D'], inplace=True)

# Remove rows with NaN values (if any)
daily_summary = daily_summary.dropna()

# Print the updated DataFrame
print(daily_summary)

# Reverse the order of rows in the DataFrame
daily_summary_reversed = daily_summary[::-1]

# Printing the reversed DataFrame
print(daily_summary_reversed)


## Correlation Between All Stocks

# Correlation
# Assuming `all_stocks_data` is the DataFrame we downloaded earlier
# Pivot the data to get closing prices of all stocks in columns
closing_prices = all_stocks_data.pivot(index='date', columns='name', values='volume')

# Calculate the correlation matrix for the closing prices
correlation_matrix = closing_prices.corr()

print("Correlation matrix between different stocks:")
print(correlation_matrix)

# Optionally, save the correlation matrix to a CSV file
correlation_matrix.to_csv('correlation_matrix_between_stocks.csv')

# Correlation for Top two Stocks

# Calculate monthly returns for each stock
data['monthly_return'] = data.groupby([data['date'].dt.year, data['date'].dt.month])['close'].pct_change()

# Group data by month and year
grouped_data = data.groupby([data['date'].dt.year, data['date'].dt.month])

# Initialize dictionary to store correlation and names of the two best performing stocks for each month and year
correlation_best_stocks = {}

# Iterate over each month and year group
for group, month_data in grouped_data:
    # Sort stocks based on monthly return
    top_stocks = month_data.groupby('name')['monthly_return'].mean().nlargest(2).tail(2).index.tolist()  # Ensure only two stocks are selected
    # Extract data for the top two performing stocks
    top_stock_data = month_data[month_data['name'].isin(top_stocks)]
    # Calculate correlation between the top two performing stocks
    correlation = top_stock_data.pivot_table(index='date', columns='name', values='monthly_return').corr().iloc[0, 1]
    # Store correlation and names of the two best performing stocks in dictionary
    stock_names = ', '.join(top_stocks)
    correlation_best_stocks[group] = {'correlation': correlation, 'stocks': stock_names}

# Print correlation and names of the two best performing stocks for each month and year
print("Correlation between the two best performing stocks for each month and year:")
for group, data in correlation_best_stocks.items():
    print(f"Month-Year: {group}, Correlation: {data['correlation']}, Stocks: {data['stocks']}")


# Calculate rolling standard deviation for all companies as a measure of volatility
data['rolling_std'] = data.groupby('name')['close'].transform(lambda x: x.rolling(window=30).std())

# Calculate the average trading volume for all companies
avg_volume = data.groupby('name')['volume'].mean().reset_index()

# Calculate the average volatility for all companies
avg_volatility = data.groupby('name')['rolling_std'].mean().reset_index()

# Merging average volume and average volatility
avg_volume_volatility = pd.merge(avg_volume, avg_volatility, on='name')

# Identifying the company with the highest average trading volume
max_volume_company = avg_volume_volatility.loc[avg_volume_volatility['volume'].idxmax()]

# Identifying the company with the highest average volatility
max_volatility_company = avg_volume_volatility.loc[avg_volume_volatility['rolling_std'].idxmax()]

# Plot for the company with the highest average trading volume
max_volume_df = data[data['name'] == max_volume_company['name']]
plt.figure(figsize=(10, 6))
plt.bar(max_volume_df['date'], max_volume_df['volume'], color='purple')
plt.title(f'Trading Volume for {max_volume_company["name"]}')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()



import matplotlib.pyplot as plt

# Identify the company with the highest average volatility
max_volatility_company = avg_volume_volatility.loc[avg_volume_volatility['rolling_std'].idxmax()]

# Filter data for the company with the highest average volatility
max_volatility_df = data[data['name'] == max_volatility_company['name']]

# Plot the rolling standard deviation (volatility) for the company
plt.figure(figsize=(10, 6))
plt.plot(max_volatility_df['date'], max_volatility_df['rolling_std'], color='red')
plt.title(f'Volatility for {max_volatility_company["name"]}')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.show()


import plotly.graph_objects as go
import pandas as pd

# Filter the DataFrame for the year 2024 and the company with the highest average trading volume
max_volume_df_2024 = max_volume_df[(max_volume_df['date'] >= pd.Timestamp('2024-01-01')) & (max_volume_df['date'] <= pd.Timestamp('2024-5-16'))]

# Create an interactive plot using Plotly
fig = go.Figure()

# Add bar chart for trading volume
fig.add_trace(go.Bar(x=max_volume_df_2024['date'], y=max_volume_df_2024['volume'], marker_color='purple'))

# Update layout for a better visual appearance
fig.update_layout(
    title=f'Trading Volume for {max_volume_company["name"]} in 2024',
    xaxis_title='Date',
    yaxis_title='Volume',
    template='plotly_white',
    xaxis_rangeslider_visible=True  # Enable the range slider for zooming
)

# Show the figure
fig.show()
