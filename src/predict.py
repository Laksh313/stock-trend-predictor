import sys
import os
import joblib
import yfinance as yf
import pandas as pd
from data_preprocessing import engineer_features

MODEL_PATH = "model/stock_predictor.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def fetch_latest_data(ticker, days=120):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    df.dropna(inplace=True)

    # flatten multi-index if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <TICKER>")
        sys.exit(1)

    ticker = sys.argv[1].upper()
    print(f"Fetching data for {ticker}...")
    df = fetch_latest_data(ticker)

    print("Engineering features...")
    df = engineer_features(df)

    print("Loading model...")
    model = load_model()

    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_50", "EMA_10", "Volatility",
        "RSI", "MACD", "MACD_Signal"
    ]

    latest_features = df[feature_cols].iloc[-1:].values
    prediction = model.predict(latest_features)[0]

    print("\n===== Prediction =====")
    print(f"Last close ({df.index[-1].date()}): {df['Close'].iloc[-1]:.2f}")
    if prediction == 1:
        print("📈 Model predicts: UP tomorrow")
    else:
        print("📉 Model predicts: DOWN tomorrow")

if __name__ == "__main__":
    main()
