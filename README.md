# Time-Series Analysis and Forecasting with Python using Stock Price Data 
## Overview
This project focuses on analyzing and forecasting stock market trends over five years using historical stock data. By leveraging Python and TimescaleDB, we aim to:
- Predict short-term stock price movements
- Understand market volatility
- Identify long-term trends

## Tools and Technologies
- **Python:** Libraries such as pandas, NumPy, sci-kit-learn, Statsmodels, and Prophet.
- **TimescaleDB:** Efficiently handles time-series data and provides advanced statistical functions.

## Project Steps
1. **Data Ingestion:** Use TimescaleDB to store and retrieve stock data.
2. **Exploratory Data Analysis (EDA):** Identify trends and patterns using Matplotlib, Seaborn, and Plotly.
3. **Model Building:** Employ machine learning models like ARIMA for forecasting.
4. **Model Evaluation:** Assess model performance using RMSE and MAE metrics.

## Dataset
The dataset includes attributes for each trading day: Date, Open, High, Low, Close, Volume, and Stock Name.

## Repository Contents
- **ARIMA.py:** Model Building and Forecasting
- **README.md:** Project Overview and Instructions
- **Timescale.sql:** Timescale Queries Used
- **Timescale_finance_data:** Retrieving stock data from TimescaleDB
- **exploratory_data_analysis.py:** Perform Statistical and Visualization on Data
- **extract.py:** Extract Recent Data of Various Stocks using Python
- **stockdataTimescale.py:** Importing Stock Data from TimescaleDB

---

Happy Forecasting
