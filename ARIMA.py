from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


# Convert the date column to datetime and set it as the index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Filter for a specific stock,
avgo_df = data[data['name'] == 'AVGO']['close'].asfreq('B')

# Check for stationarity
result = adfuller(avgo_df.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If the series is not stationary, differencing might be necessary

# Fit the ARIMA model
model = ARIMA(avgo_df, order=(5,1,0))  # Example order, needs optimization
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())


# Perform differencing
differenced_series = avgo_df.diff().dropna()  # Compute the difference and remove NaN values

# Check for stationarity after differencing
result_diff = adfuller(differenced_series)
print('ADF Statistic after differencing:', result_diff[0])
print('p-value after differencing:', result_diff[1])


# Fit the ARIMA model to the differenced series
model_diff = ARIMA(differenced_series, order=(2, 1, 2))  # Example order, needs optimization
model_fit_diff = model_diff.fit()

# Summary of the differenced model
print(model_fit_diff.summary())


# Let's plot forecasting model plots
def display_model_plots(model, co_name):
    plt.style.use('fivethirtyeight')
    model.plot_diagnostics(figsize=(20, 15))
    plt.suptitle(f'Model diagnostics of {co_name}', fontsize=25)
    plt.subplots_adjust(top=0.93)
    plt.show()
    plt.style.use('default')

# Display model plots for AAPL
display_model_plots(model_fit_diff, 'AVGO')


from sklearn.metrics import mean_squared_error
import numpy as np

# Make predictions on the training data
train_predictions_diff = model_fit_diff.predict(start=differenced_series.index[1], end=differenced_series.index[-1])

# Align the indices of the differenced series and predictions
train_predictions_diff = train_predictions_diff[differenced_series.index[1]:]

# Calculate RMSE
rmse_diff = np.sqrt(mean_squared_error(differenced_series[1:], train_predictions_diff))

print("RMSE:", rmse_diff)


from sklearn.metrics import mean_absolute_error

# Calculate MAE
mae_diff = mean_absolute_error(differenced_series[1:], train_predictions_diff)

print("MAE:", mae_diff)
