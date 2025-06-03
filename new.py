
# Stock Price Prediction using Linear Regression with SMA and RSI features

import pandas as pd  # For data handling
import numpy as np  # For numerical computations
from sklearn.preprocessing import MinMaxScaler  # For data normalization
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.linear_model import LinearRegression  # For regression modeling
from sklearn.metrics import mean_squared_error, r2_score  # For performance metrics
import matplotlib.pyplot as plt  # For visualization


# Function to calculate Relative Strength Index (RSI)
def compute_rsi(data, window=14):
    """
    Computes the Relative Strength Index (RSI) for a given price series.

    Parameters:
        data (pd.Series): Series of prices.
        window (int): Look-back window for RSI calculation.

    Returns:
        pd.Series: RSI values.
    """
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Load stock data
df = pd.read_csv('stock_data.csv')

# Calculate technical indicators
df['SMA'] = df['Open'].rolling(window=20).mean()  # Simple Moving Average (20-day)
df['RSI'] = compute_rsi(df['Open'], window=14)  # Relative Strength Index (14-day)

# Normalize features: 'Open', 'SMA', 'RSI'
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['Open', 'SMA', 'RSI']].dropna())

# Create input features (X) and target variable (y)
X = df_scaled[:-1]  # All rows except the last
y = df_scaled[1:, 0]  # Next day's normalized 'Open' price

# Ensure there is sufficient data
if len(X) == 0 or len(y) == 0:
    print("Not enough data after preprocessing and splitting. Cannot train the model.")
else:
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'R-squared: {r2}')

    # Plot actual vs predicted normalized prices
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(y_test):], y_test, label='Actual')
    plt.plot(df.index[-len(y_test):], y_pred, label='Predicted')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()



