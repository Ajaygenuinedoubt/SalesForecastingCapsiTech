# -*- coding: utf-8 -*-
"""Sales_forecasting.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1F1wm_OGsykWR70lTfGX1d8ikKTVmXGMZ

**1.Install required library are as following:-**



*   Pandas

*   Prophet
* Matplotlib
* Numpy
* Scipy
* Ipython
"""

!pip install numpy pandas matplotlib seaborn scikit-learn statsmodels tensorflow

import pandas as pd

"""2.Upoad CSV file for PreProcessing"""

df=pd.read_csv("/content/Dataset.csv")
df.head(5)

# Convert the 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set 'Order Date' as the index for time series analysis
df.set_index('Order Date', inplace=True)

# Aggregate the data by month to calculate the total revenue or profit per month
monthly_sales = df.resample('M').sum()['Total Revenue']

# Display the first few rows of the aggregated data
monthly_sales.head()

"""**3.Visualize the Graph Between Monthly Sale and Total revenue**"""

import matplotlib.pyplot as plt

# Plot the monthly sales to visualize the trend
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales, label='Total Revenue')
plt.title('Monthly Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.grid(True)
plt.legend()
plt.show()

"""***4.Graph for Total Revenue ,Trend And Seasonal for visualization***"""

from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose the time series into trend, seasonal, and residual components
decomposition = seasonal_decompose(monthly_sales, model='additive')

# Plot the decomposed components
decomposition.plot()
plt.show()

"""**4.Install this library for forecasting**"""

!pip install pmdarima

"""5. Here we train the model and forecasting the prices
* Blue colour Actual Prediction
* Red  Prediction
"""

# Install necessary libraries (uncomment in Google Colab)
# !pip install pmdarima

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and preprocess the dataset
df = pd.read_csv('/content/Dataset.csv')  # Update with your dataset path
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)

# Aggregate the data by month (for simplicity)
monthly_sales = df.resample('M').sum()['Total Revenue']

# Split the data into training and testing sets (80% train, 20% test)
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales[:train_size], monthly_sales[train_size:]

# Fit the ARIMA model using auto_arima to find the best parameters (p, d, q)
stepwise_model = auto_arima(train, seasonal=False, trace=True, suppress_warnings=True)
stepwise_model.summary()

# Fit ARIMA model based on auto_arima suggestions
arima_model = ARIMA(train, order=stepwise_model.order)
arima_result = arima_model.fit()

# Forecast future values (length of the test set)
forecast = arima_result.forecast(steps=len(test))

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Sales', color='blue')
plt.plot(test.index, forecast, label='Forecasted Sales', color='red')
plt.title('Actual vs Forecasted Sales')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate the model
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# Forecast future sales for the next 12 months
future_forecast = arima_result.forecast(steps=12)
print("Future Sales Prediction (next 12 months):\n", future_forecast)

"""**5.Model Training LSTM the the graph look **"""

# Install necessary libraries (uncomment if running in Google Colab)
# !pip install tensorflow
# !pip install scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess the dataset
df = pd.read_csv('/content/Dataset.csv')  # Update with your dataset path

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set 'Order Date' as index
df.set_index('Order Date', inplace=True)

# Check if 'Total Revenue' exists and is numerical
if 'Total Revenue' in df.columns:
    print(f"Total Revenue column found: {df['Total Revenue'].dtype}")

    # Aggregate the data by month (resample and sum the revenue)
    monthly_sales = df['Total Revenue'].resample('M').sum()

    # Plot the sales data
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales, label='Total Revenue')
    plt.title('Monthly Total Revenue')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Scale the data (LSTM models perform better with scaled data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    monthly_sales_scaled = scaler.fit_transform(monthly_sales.values.reshape(-1, 1))

    # Split the data into training and testing sets (80% train, 20% test)
    train_size = int(len(monthly_sales_scaled) * 0.8)
    train, test = monthly_sales_scaled[:train_size], monthly_sales_scaled[train_size:]

    # Function to create sequences for LSTM
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(x), np.array(y)

    seq_length = 12  # 12 months of data to predict the next month
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)

    # Reshape the data for LSTM (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions on the test set
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)  # Rescale the predictions back to the original scale

    # Rescale the test set for comparison
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label='Actual Sales', color='blue')
    plt.plot(predictions_rescaled, label='Predicted Sales', color='red')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the model
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    print(f'RMSE: {rmse}')

    # Forecast future sales for the next 12 months
    last_seq = test[-seq_length:]  # Take the last sequence from the test data
    future_predictions = []
    for _ in range(15):
        last_seq = last_seq.reshape((1, seq_length, 1))
        next_pred = model.predict(last_seq)[0]
        future_predictions.append(next_pred)
        last_seq = np.append(last_seq[:, 1:, :], [[next_pred]], axis=1)

    # Rescale the future predictions back to the original scale
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)

    print("Future Sales Prediction (next 12 months):\n", future_predictions_rescaled)

else:
    print("'Total Revenue' column not found or not numerical.")

"""**6.Trend,Resid,Seasonal And Total revenue graph**"""

# Install necessary libraries (uncomment if running in Google Colab)
# !pip install tensorflow
# !pip install scikit-learn
# !pip install statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess the dataset
df = pd.read_csv('/content/Dataset.csv')  # Update with your dataset path

# Convert 'Order Date' to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Set 'Order Date' as index
df.set_index('Order Date', inplace=True)

# Check if 'Total Revenue' exists and is numerical
if 'Total Revenue' in df.columns:
    print(f"Total Revenue column found: {df['Total Revenue'].dtype}")

    # Aggregate the data by month (resample and sum the revenue)
    monthly_sales = df['Total Revenue'].resample('M').sum()

    # Perform seasonal decomposition to understand trend and seasonality
    decomposition = seasonal_decompose(monthly_sales, model='additive')
    decomposition.plot()
    plt.show()

    # Plot the sales data
    plt.figure(figsize=(10, 6))
    plt.plot(monthly_sales, label='Total Revenue')
    plt.title('Monthly Total Revenue')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Scale the data (LSTM models perform better with scaled data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    monthly_sales_scaled = scaler.fit_transform(monthly_sales.values.reshape(-1, 1))

    # Split the data into training and testing sets (80% train, 20% test)
    train_size = int(len(monthly_sales_scaled) * 0.8)
    train, test = monthly_sales_scaled[:train_size], monthly_sales_scaled[train_size:]

    # Function to create sequences for LSTM
    def create_sequences(data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(x), np.array(y)

    seq_length = 12  # 12 months of data to predict the next month
    X_train, y_train = create_sequences(train, seq_length)
    X_test, y_test = create_sequences(test, seq_length)

    # Reshape the data for LSTM (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build the LSTM model with more layers
    model = Sequential()
    model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions on the test set
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions)  # Rescale the predictions back to the original scale

    # Rescale the test set for comparison
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_rescaled, label='Actual Sales', color='blue')
    plt.plot(predictions_rescaled, label='Predicted Sales', color='red')
    plt.title('Actual vs Predicted Sales')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate the model with multiple metrics
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

    # Forecast future sales for the next 15 months (extended horizon)
    last_seq = test[-seq_length:]  # Take the last sequence from the test data
    future_predictions = []
    for _ in range(15):
        last_seq = last_seq.reshape((1, seq_length, 1))
        next_pred = model.predict(last_seq)[0]
        future_predictions.append(next_pred)
        last_seq = np.append(last_seq[:, 1:, :], [[next_pred]], axis=1)

    # Rescale the future predictions back to the original scale
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)

    # Plot future sales predictions
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions_rescaled, label='Future Sales Prediction', color='green')
    plt.title('Future Sales Prediction (Next 15 Months)')
    plt.xlabel('Month')
    plt.ylabel('Total Revenue')
    plt.legend()
    plt.grid(True)
    plt.show()

else:
    print("'Total Revenue' column not found or not numerical.")

"""**7.Our Model predict the cost of price for 12 months in lower value as well as High value**"""

# Install necessary libraries (uncomment in Google Colab)
# !pip install prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your dataset (ensure the correct path)
df = pd.read_csv('/content/Dataset.csv')  # Update with your dataset path

# Convert 'Order Date' to datetime and set as index
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)

# Ensure 'Total Revenue' exists and handle missing values
if 'Total Revenue' in df.columns:
    df['Total Revenue'] = df['Total Revenue'].fillna(0)  # Handle missing values

    # Prepare the data for Prophet (it requires a specific column format)
    df_prophet = df['Total Revenue'].resample('M').sum().reset_index()
    df_prophet.columns = ['ds', 'y']  # Prophet requires 'ds' for date and 'y' for the target

    # Initialize Prophet model and add yearly seasonality (if appropriate)
    prophet_model = Prophet(yearly_seasonality=True)

    # Fit the model
    prophet_model.fit(df_prophet)

    # Make future dataframe for predictions (forecast for the next 12 months)
    future = prophet_model.make_future_dataframe(periods=12, freq='M')

    # Predict future values
    forecast = prophet_model.predict(future)

    # Plot the forecasted data
    plt.figure(figsize=(10, 6))
    prophet_model.plot(forecast)
    plt.title('Sales Forecast (Next 12 Months)')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.grid(True)
    plt.show()

    # Optional: Plot components to understand the trends and seasonality
    plt.figure(figsize=(10, 6))
    prophet_model.plot_components(forecast)
    plt.show()

    # Show the forecast for the next 12 months
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
    print("Future Sales Prediction (Next 12 Months):")
    print(future_forecast)

else:
    print("'Total Revenue' column not found or not numerical.")

# Install necessary libraries (uncomment in Google Colab)
# !pip install prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from IPython.display import HTML
import base64

# Load your dataset (ensure the correct path)
df = pd.read_csv('/content/Dataset.csv')  # Update with your dataset path

# Convert 'Order Date' to datetime and set as index
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)

# Ensure 'Total Revenue' exists and handle missing values
if 'Total Revenue' in df.columns:
    df['Total Revenue'] = df['Total Revenue'].fillna(0)  # Handle missing values

    # Prepare the data for Prophet (it requires a specific column format)
    df_prophet = df['Total Revenue'].resample('M').sum().reset_index()
    df_prophet.columns = ['ds', 'y']  # Prophet requires 'ds' for date and 'y' for the target

    # Initialize Prophet model and add yearly seasonality (if appropriate)
    prophet_model = Prophet(yearly_seasonality=True)

    # Fit the model
    prophet_model.fit(df_prophet)

    # Make future dataframe for predictions (forecast for the next 12 months)
    future = prophet_model.make_future_dataframe(periods=12, freq='M')

    # Predict future values
    forecast = prophet_model.predict(future)

    # Plot the forecasted data
    plt.figure(figsize=(10, 6))
    prophet_model.plot(forecast)
    plt.title('Sales Forecast (Next 12 Months)')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.grid(True)
    plt.show()

    # Optional: Plot components to understand the trends and seasonality
    plt.figure(figsize=(10, 6))
    prophet_model.plot_components(forecast)
    plt.show()

    # Show the forecast for the next 12 months
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)
    print("Future Sales Prediction (Next 12 Months):")
    print(future_forecast)

    # Save the forecast to a CSV file
    future_forecast.to_csv('sales_forecast.csv', index=False)

    # Create a download button
    def create_download_link(df, title="Download CSV file", filename="data.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'
        return HTML(href)

    # Display the download link
    create_download_link(future_forecast, title="Click here to download the forecasted data", filename="sales_forecast.csv")

else:
    print("'Total Revenue' column not found or not numerical.")

!pip install xlsxwriter

"""**8.For plot the graph of Total revenue using Matplot3D**"""

# Required Libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from IPython.display import HTML
import base64
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.cm as cm  # For colormap

# Load the dataset
df = pd.read_csv('/content/Dataset.csv')

# Step 1: Data Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Ensure 'Total Revenue' exists and handle missing values if any
if 'Total Revenue' in df.columns:
    df['Total Revenue'] = df['Total Revenue'].fillna(0)  # Handle missing values

    # Remove Outliers in 'Total Revenue' using Z-score method
    z_scores = np.abs(stats.zscore(df['Total Revenue']))
    df = df[(z_scores < 3)]  # Remove rows where z-score is greater than 3 (outliers)

    # Aggregate data by month for Prophet model
    df_monthly = df.resample('M', on='Order Date').sum()

    # Apply Log Transformation to 'Total Revenue'
    df_monthly['Total Revenue'] = np.log1p(df_monthly['Total Revenue'])

    # Prepare data for Prophet model ('ds' for date and 'y' for target)
    df_prophet = df_monthly.reset_index()[['Order Date', 'Total Revenue']]
    df_prophet.columns = ['ds', 'y']

    # Initialize Prophet model with yearly seasonality
    prophet_model = Prophet(yearly_seasonality=True)

    # Fit the model
    prophet_model.fit(df_prophet)

    # Make future predictions for the next 12 months
    future = prophet_model.make_future_dataframe(periods=12, freq='M')
    forecast = prophet_model.predict(future)

    # Inverse the log transformation to revert back to original scale
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # Step 3: Download predicted values as CSV
    def download_link(df, title="Download CSV file", filename="predictions.csv"):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title}</a>'
        return HTML(href)

    # Extract relevant columns
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

    # Provide the download link for the predicted values
    display(download_link(future_forecast))

    # Step 4: Plot the forecast
    plt.figure(figsize=(10, 6))
    prophet_model.plot(forecast)
    plt.title('Future Sales Forecast (Next 12 Months)')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.grid(True)
    plt.show()

    # Step 5: Plot 3D graph of forecasted values with colors
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Convert dates to numbers for 3D plotting
    date_nums = pd.to_datetime(forecast['ds']).map(pd.Timestamp.toordinal)

    # Set 3D axes data
    X = date_nums
    Y = forecast['yhat']
    Z = forecast['yhat_upper'] - forecast['yhat_lower']  # Uncertainty range

    # Normalize Z for colormap scaling
    norm = plt.Normalize(Z.min(), Z.max())

    # Use a colormap (e.g., 'plasma')
    colors = cm.plasma(norm(Z))

    # 3D Scatter plot with color mapping based on Z values (uncertainty)
    ax.scatter(X, Y, Z, c=colors, cmap='plasma', s=60)

    # Set axis labels
    ax.set_xlabel('Date (Ordinal)')
    ax.set_ylabel('Forecasted Revenue')
    ax.set_zlabel('Prediction Interval (Uncertainty)')

    # Set title
    ax.set_title('Colorful 3D Forecast of Total Revenue with Uncertainty')

    # Show the plot
    plt.show()

else:
    print("'Total Revenue' column not found or not numerical.")

"""9.Adding some extra features to visualize the graph using

* Bolinger Band
* Rolling Means
* Moving Average
* Lower scale
* Higher Scale


10. Add Downloadable button to download Prediction data in this format

* CSV
* Json
* XLSX
"""

# Required Libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from IPython.display import HTML
import base64
import io
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('/content/Dataset.csv')

# Step 1: Data Preprocessing
# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Ensure 'Total Revenue' exists and handle missing values if any
if 'Total Revenue' in df.columns:
    df['Total Revenue'] = df['Total Revenue'].fillna(0)  # Handle missing values

    # Remove Outliers in 'Total Revenue' using Z-score method
    z_scores = np.abs(stats.zscore(df['Total Revenue']))
    df = df[(z_scores < 3)]  # Remove rows where z-score is greater than 3 (outliers)

    # Aggregate data by month for Prophet model
    df_monthly = df.resample('M', on='Order Date').sum()

    # Apply Log Transformation to 'Total Revenue' to handle skewness and prevent negative values
    df_monthly['Total Revenue'] = np.log1p(df_monthly['Total Revenue'])

    # Prepare data for Prophet model ('ds' for date and 'y' for target)
    df_prophet = df_monthly.reset_index()[['Order Date', 'Total Revenue']]
    df_prophet.columns = ['ds', 'y']

    # Initialize Prophet model with yearly seasonality
    prophet_model = Prophet(yearly_seasonality=True)

    # Fit the model
    prophet_model.fit(df_prophet)

    # Make future predictions for the next 12 months
    future = prophet_model.make_future_dataframe(periods=12, freq='M')
    forecast = prophet_model.predict(future)

    # Inverse the log transformation to revert back to original scale
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # Extract relevant columns (ds: date, yhat: predicted, yhat_lower and yhat_upper: uncertainty intervals)
    future_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

    # Step 2: Function to create download link for CSV, JSON, and XLSX formats
    def download_link(df, title="Download file", filename="predictions.csv", file_format="csv"):
        if file_format == 'csv':
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
            href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{title} (CSV)</a>'
        elif file_format == 'json':
            json_data = df.to_json(orient='records')
            b64 = base64.b64encode(json_data.encode()).decode()  # Encode to base64
            href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{title} (JSON)</a>'
        elif file_format == 'xlsx':
            towrite = io.BytesIO()
            df.to_excel(towrite, index=False, engine='xlsxwriter')  # Write to Excel file in memory
            towrite.seek(0)
            b64 = base64.b64encode(towrite.read()).decode()  # Encode to base64
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{title} (XLSX)</a>'
        return HTML(href)

    # Provide download links for CSV, JSON, and XLSX
    display(download_link(future_forecast, filename="predictions.csv", file_format="csv"))
    display(download_link(future_forecast, filename="predictions.json", file_format="json"))
    display(download_link(future_forecast, filename="predictions.xlsx", file_format="xlsx"))

    # Step 3: Plot Candlestick chart with Bollinger Bands

    # Add Bollinger Bands
    window = 3  # Choose window size (e.g., 3 months for a smoother view)
    forecast['rolling_mean'] = forecast['yhat'].rolling(window=window).mean()
    forecast['rolling_std'] = forecast['yhat'].rolling(window=window).std()

    # Calculate upper and lower Bollinger Bands
    forecast['Bollinger_upper'] = forecast['rolling_mean'] + (forecast['rolling_std'] * 2)
    forecast['Bollinger_lower'] = forecast['rolling_mean'] - (forecast['rolling_std'] * 2)

    # Step 4: Plot Candlestick chart using Plotly

    fig = go.Figure(data=[go.Candlestick(x=forecast['ds'],
                                         open=forecast['yhat_lower'],
                                         high=forecast['yhat_upper'],
                                         low=forecast['yhat_lower'],
                                         close=forecast['yhat'],
                                         name='Candlestick')])

    # Add Bollinger Bands to the plot
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Bollinger_upper'],
                             line=dict(color='blue', width=1), name='Upper Band'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Bollinger_lower'],
                             line=dict(color='red', width=1), name='Lower Band'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['rolling_mean'],
                             line=dict(color='green', width=2), name='Moving Average'))

    # Update layout
    fig.update_layout(title='Candlestick Chart with Bollinger Bands',
                      xaxis_title='Date',
                      yaxis_title='Predicted Total Revenue',
                      template='plotly_dark')

    # Show the plot
    fig.show()

    # Step 5: Display future forecast data
    print("Future Sales Prediction (Next 12 Months):")
    print(future_forecast)

else:
    print("'Total Revenue' column not found or not numerical.")

# Required Libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from IPython.display import HTML
import base64
import io
import plotly.graph_objects as go
import plotly.express as px

# Load the dataset
df = pd.read_csv('/content/Dataset.csv')

# Step 1: Data Preprocessing
# Convert 'Order Date' to datetime format
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Ensure 'Total Revenue' exists and handle missing values if any
if 'Total Revenue' in df.columns:
    df['Total Revenue'] = df['Total Revenue'].fillna(0)  # Handle missing values

    # Remove Outliers in 'Total Revenue' using Z-score method
    z_scores = np.abs(stats.zscore(df['Total Revenue']))
    df = df[(z_scores < 3)]  # Remove rows where z-score is greater than 3 (outliers)

    # Aggregate data by month for Prophet model
    df_monthly = df.resample('M', on='Order Date').sum()

    # Apply Log Transformation to 'Total Revenue' to handle skewness and prevent negative values
    df_monthly['Total Revenue'] = np.log1p(df_monthly['Total Revenue'])

    # Prepare data for Prophet model ('ds' for date and 'y' for target)
    df_prophet = df_monthly.reset_index()[['Order Date', 'Total Revenue']]
    df_prophet.columns = ['ds', 'y']

    # Initialize Prophet model with yearly seasonality
    prophet_model = Prophet(yearly_seasonality=True)

    # Fit the model
    prophet_model.fit(df_prophet)

    # Make future predictions for the next 12 months
    future = prophet_model.make_future_dataframe(periods=12, freq='M')
    forecast = prophet_model.predict(future)

    # Inverse the log transformation to revert back to original scale
    forecast['yhat'] = np.expm1(forecast['yhat'])
    forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
    forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])

    # Step 3: Plot Candlestick chart with Bollinger Bands

    # Add Bollinger Bands
    window = 3  # Choose window size (e.g., 3 months for a smoother view)
    forecast['rolling_mean'] = forecast['yhat'].rolling(window=window).mean()
    forecast['rolling_std'] = forecast['yhat'].rolling(window=window).std()

    # Calculate upper and lower Bollinger Bands
    forecast['Bollinger_upper'] = forecast['rolling_mean'] + (forecast['rolling_std'] * 2)
    forecast['Bollinger_lower'] = forecast['rolling_mean'] - (forecast['rolling_std'] * 2)

    # Step 4: Plot Candlestick chart using Plotly

    fig = go.Figure(data=[go.Candlestick(x=forecast['ds'],
                                         open=forecast['yhat_lower'],
                                         high=forecast['yhat_upper'],
                                         low=forecast['yhat_lower'],
                                         close=forecast['yhat'],
                                         name='Candlestick')])

    # Add Bollinger Bands to the plot
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Bollinger_upper'],
                             line=dict(color='blue', width=1), name='Upper Band'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['Bollinger_lower'],
                             line=dict(color='red', width=1), name='Lower Band'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['rolling_mean'],
                             line=dict(color='green', width=2), name='Moving Average'))

    # Update layout
    fig.update_layout(title='Candlestick Chart with Bollinger Bands',
                      xaxis_title='Date',
                      yaxis_title='Predicted Total Revenue',
                      template='plotly_dark')

    # Show the plot
    fig.show()

    # Step 5: Plot 3D Surface Plot
    # Create meshgrid for 3D plot
    X = np.arange(len(forecast['ds']))
    Y = np.array([forecast['yhat'], forecast['yhat_upper'], forecast['yhat_lower']])
    Z = np.array([forecast['yhat'], forecast['yhat_upper'], forecast['yhat_lower']])

    fig_3d = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])

    fig_3d.update_layout(title='3D Surface Plot of Forecasted Data',
                         scene=dict(
                             xaxis_title='Time',
                             yaxis_title='Yhat Values',
                             zaxis_title='Forecasts'),
                         autosize=True)

    fig_3d.show()

else:
    print("'Total Revenue' column not found or not numerical.")

