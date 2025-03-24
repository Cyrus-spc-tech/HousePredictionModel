import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.title("Stock Price Prediction")

# User input for stock ticker
ticker = st.text_input("Enter the stock ticker symbol (e.g., AAPL for Apple):", "AAPL")

# Fetch stock data
stock_data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Check if data is available
if stock_data.empty:
    st.write(f"No data found for ticker symbol: {ticker}")
else:
    # Display stock data
    st.write(f"Stock data for {ticker}:")
    st.write(stock_data.tail())

    # Prepare data for prediction
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Date'] = stock_data['Date'].map(pd.Timestamp.toordinal)

    X = stock_data[['Date']]
    y = stock_data['Close']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Check if the split was successful
    if len(X_train) == 0 or len(X_test) == 0:
        st.write("Not enough data to split into training and testing sets. Please try a different ticker or date range.")
    else:
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")

        # Plot actual vs predicted prices
        fig, ax = plt.subplots()
        ax.plot(stock_data.index, stock_data['Close'], label='Actual Prices')
        ax.plot(X_test.index, y_pred, label='Predicted Prices', linestyle='--')
        ax.set_xlabel("Date")
        ax.set_ylabel("Stock Price")
        ax.set_title(f"Actual vs Predicted Stock Prices for {ticker}")
        ax.legend()
        st.pyplot(fig)

        # Predict future stock price
        future_date = st.date_input("Select a future date for prediction:", pd.to_datetime("2023-01-01"))
        future_date_ordinal = pd.Timestamp(future_date).toordinal()
        future_price = model.predict([[future_date_ordinal]])

        st.write(f"Predicted stock price for {ticker} on {future_date}: ${future_price[0]:.2f}")