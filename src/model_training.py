from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(df, model_path):
    feature_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_10", "SMA_50", "EMA_10", "Volatility",
        "RSI", "MACD", "MACD_Signal"
    ]

    for col in feature_cols + ["Target"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    X = df[feature_cols]
    y = df["Target"]

    X = X.dropna()
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained. Test Accuracy: {acc:.2%}")

    os.makedirs("model", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
