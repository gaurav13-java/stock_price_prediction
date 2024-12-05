import streamlit as st
from utils import download_stock_data
from model import add_features, preprocess_data, train_model, evaluate_model
import pandas as pd

st.title("Stock Price Prediction App")

# Input fields
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL)", value="AAPL")
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
    st.write(f"### Model Evaluation: MSE = {mse}")

    # Step 6: Visualize predictions
    st.line_chart(pd.DataFrame({
        "Actual": y_test.values.flatten(),
        "Predicted": predictions.flatten()
    }))

# Visualize predictions in a cleaner way
predictions_df = pd.DataFrame({
    "Date": data.iloc[len(data) - len(y_test):]["Date"],
    "Actual Price": y_test.values.flatten(),
    "Predicted Price": predictions.flatten()
})

# Display as a table
st.write("### Actual vs Predicted Stock Prices")
st.dataframe(predictions_df)

# Optionally, add a line chart for better visualization
st.line_chart(predictions_df.set_index('Date'))

