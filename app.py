import streamlit as st
from utils import download_stock_data
from model import add_features, preprocess_data, train_model, evaluate_model
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
# URL for background image
background_image_url = "https://images.unsplash.com/photo-1584697964156-e1e7b9d0a9bc"

# HTML and inline CSS to apply the background image
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url({background_image_url}) no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
    }}
    .sidebar {{
        background: rgba(255, 255, 255, 0.8);
    }}
    .main {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    """, unsafe_allow_html=True
)

st.title("Stock Price Prediction App")

# Input fields
tickers = ['GME', 'LULU', 'UBER', 'SOUN','ULTA', 'TSLA','AAPL','PATH','HPE','INTC','LYFT']
ticker = st.selectbox("Enter Stock Ticker Symbol", tickers)
start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))

if st.button("Predict"):
    # Step 1: Download stock data
    data = download_stock_data(ticker, start_date, end_date)
    st.write("### Stock Data", data.tail())

    # Step 2: Feature engineering
    data = add_features(data)
    st.write("### Data with Features", data.tail())

    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Step 4: Train model
    model = train_model(X_train, y_train)

    # Step 5: Evaluate model
    mse, predictions = evaluate_model(model, X_test, y_test)

    # Calculate percentage difference
    diff = ((predictions.flatten() - y_test.values.flatten()) / y_test.values.flatten()) * 100
    avg_diff = diff.mean()

    # Display metrics
    st.write("### Model Evaluation Metrics")
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**Average Percentage Difference:** {avg_diff:.2f}%")

    # Create a predictions dataframe
    predictions_df = pd.DataFrame({
        "Date": data.iloc[len(data) - len(y_test):]["Date"],
        "Actual Price": y_test.values.flatten(),
        "Predicted Price": predictions.flatten()
    })

    # Display predictions as a table
    st.write("### Actual vs Predicted Stock Prices")
    st.dataframe(predictions_df)

    # Plotly interactive chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Actual Price'],
                             mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted Price'],
                             mode='lines', name='Predicted', line=dict(color='red', dash='dot')))
    fig.update_layout(title="Actual vs Predicted Stock Prices",
                      xaxis_title="Date", yaxis_title="Price", template="plotly_dark")
    st.plotly_chart(fig)
