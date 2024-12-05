import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def add_features(data):
    """
    Adds technical indicators like SMA and RSI to the dataset.
    """
    data['SMA_14'] = data['Close'].rolling(window=14).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].pct_change().rolling(14).mean()))
    data.dropna(inplace=True)
    return data

def preprocess_data(data):
    """
    Splits data into features (X) and target (y), and applies scaling.
    """
    features = ['Open', 'High', 'Low', 'Volume', 'SMA_14', 'RSI']
    target = ['Close']

    X = data[features]
    y = data[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains a linear regression model on the training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and returns the mean squared error and predictions.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions
