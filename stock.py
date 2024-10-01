import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# 1. Fetch Stock Data
def get_stock_data(ticker, period='10y'):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

# 2. Preprocess Data (Technical Indicators)
def compute_technical_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data.dropna(inplace=True)
    return data

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 3. Build PyTorch Model for Stock Prediction
class StockPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(StockPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 4. Prepare Data for Training
def prepare_data(data):
    features = ['SMA_20', 'SMA_50', 'RSI']
    target = 'Close'
    
    # Normalize data
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    X = data[features].values
    y = data[target].values
    
    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    return X, y, scaler

# 5. Train the Model
def train_model(model, X_train, y_train, epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    print("Model trained successfully!")
    return model

# 6. Save Model
def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

# 7. Load Model
def load_model(file_name, input_size):
    model = StockPredictionModel(input_size)
    model.load_state_dict(torch.load(file_name))
    model.eval()
    print(f"Model loaded from {file_name}")
    return model

# 8. Plot Stock Data
def plot_stock_data(data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label=f'{ticker} Closing Price')
    plt.plot(data['SMA_20'], label='20-day SMA')
    plt.plot(data['SMA_50'], label='50-day SMA')
    plt.title(f'{ticker} Stock Price and Moving Averages')
    plt.legend()
    plt.show()

# 9. Predict Future Price Movement
def predict(model, X_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
    return predictions

# 10. Investment Recommendation
def investment_recommendation(prediction, last_close_price):
    if prediction > last_close_price:
        return "Buy: The stock is predicted to rise."
    else:
        return "Sell: The stock is predicted to fall or stagnate."

# 11. Main Program
def main(ticker):
    # Step 1: Get stock data
    print(f"Fetching data for {ticker}...")
    data = get_stock_data(ticker)
    
    # Step 2: Compute technical indicators
    data = compute_technical_indicators(data)
    
    # Step 3: Prepare data for training
    X, y, scaler = prepare_data(data)
    
    # Step 4: Train the model
    input_size = X.shape[1]
    model = StockPredictionModel(input_size)
    trained_model = train_model(model, X, y, epochs=100)
    
    # Step 5: Save the model
    save_model(trained_model, f'{ticker}_stock_model.pth')
    
    # Step 6: Visualize stock data
    plot_stock_data(data, ticker)
    
    # Step 7: Predict the next day's price
    last_day_features = X[-1].view(1, -1)  # Take the last day's features
    prediction = predict(trained_model, last_day_features)
    last_close_price = data['Close'].values[-1]
    
    # Step 8: Provide an investment recommendation
    recommendation = investment_recommendation(prediction.item(), last_close_price)
    
    print(f"Last close price: {last_close_price}")
    print(f"Predicted next day's price: {prediction.item()}")
    print(f"Investment Recommendation: {recommendation}")

if __name__ == "__main__":
    main(ticker='AAPL')  # You can change the ticker to any stock
