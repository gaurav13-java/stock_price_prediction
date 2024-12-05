import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    """
    Downloads stock data for a given ticker symbol and date range.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.
    
    Returns:
        pd.DataFrame: Stock data.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data
