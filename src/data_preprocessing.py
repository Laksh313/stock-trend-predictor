# clean the data and prepare it for training
import pandas as pd
import ta

def load_and_clean(data_path):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    return df

def engineer_features(df):
    df = df.copy()

    # Flatten if multi-index from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Ensure numeric OHLCV
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame) or df[col].ndim > 1:
                df[col] = df[col].iloc[:, 0]
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Technical indicators
    df["Return"] = df["Close"].pct_change()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["Volatility"] = df["Return"].rolling(10).std()

    close = pd.Series(df["Close"].values, index=df.index)
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()

    # Target: 1 if next day close is higher, else 0
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df.dropna(inplace=True)
    return df
