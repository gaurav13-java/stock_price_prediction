import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Title of the app
st.title('Stock Price Prediction')

# Description of the app
st.write("""
This app predicts stock prices using historical data. You can select a stock ticker symbol from the dropdown list or enter your own custom symbol.
""")

# Predefined list of stock ticker symbols
tickers = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'META', 'NFLX']

# Dropdown menu for selecting a stock ticker
selected_ticker = st.selectbox('Select Stock Ticker Symbol', tickers)

# Text input field for custom stock ticker
custom_ticker = st.text_input("Or enter a custom stock ticker symbol (e.g., AAPL)")

# If the user enters a custom ticker, override the selected ticker
if custom_ticker:
    selected_ticker = custom_ticker.upper()  # Convert input to uppercase

# Display the selected ticker
st.write(f'You selected: {selected_ticker}')

# Fetch stock data based on the selected ticker
@st.cache
def get_data(ticker):
    """Function to fetch stock data"""
    data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    return data

# Fetch stock data
stock_data = get_data(selected_ticker)

# Display stock data
st.write(f"Stock Data for {selected_ticker} from Yahoo Finance")
st.write(stock_data)

# Plot the stock's closing price over time
fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Close Price'))
fig.update_layout(title=f'{selected_ticker} Stock Price Over Time', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig)

# Prepare the data for prediction (use 'Open' price as an example feature)
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].map(pd.Timestamp.to_julian_date)
X = stock_data[['Date']]  # Using 'Date' as a feature
y = stock_data['Close']  # Using 'Close' price as the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict stock prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
st.write(f'Mean Squared Error (MSE) of the model: {mse:.2f}')

# Plot predicted vs actual stock prices
fig_pred = go.Figure()

# Actual values
fig_pred.add_trace(go.Scatter(x=X_test.index, y=y_test, mode='lines', name='Actual Price'))

# Predicted values
fig_pred.add_trace(go.Scatter(x=X_test.index, y=y_pred, mode='lines', name='Predicted Price', line=dict(dash='dash')))

fig_pred.update_layout(title=f'{selected_ticker} Actual vs Predicted Stock Prices', xaxis_title='Date', yaxis_title='Price (USD)')
st.plotly_chart(fig_pred)

# Show the predicted values in a table
predicted_data = pd.DataFrame({'Date': X_test.index, 'Actual Price': y_test, 'Predicted Price': y_pred})
st.write(predicted_data)

# Display a message
st.write("""
The model uses a simple Linear Regression approach to predict stock prices based on the historical data. 
It uses the date as the feature and predicts the closing stock price.
""")
