#clean the data and prepare it for train the model
import pandas as pd

def load_and_clean(data_path):
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(df.dtypes)
    print('before dropna:', len(df))
    df.dropna(inplace=True)
    print('after dropna:', len(df))
    df["Target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df

def engineer_features(df):
    df["Daily_Return"] = (df["Close"] - df["Open"]) / df["Open"]
    df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Low"]
    df["Volume_Change"] = df["Volume"].pct_change()

    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["Volatility_5"] = df["Close"].rolling(window=5).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(4)

    df["Target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)
    return df
