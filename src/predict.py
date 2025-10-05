import sys
import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
from tensorflow.keras.models import load_model

MODEL_PATH = "model/stock_predictor.pkl"

class StockPredictor:
    """Unified predictor for all model types"""
    
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.model_type = None
        self.feature_cols = None
        self.lstm_model = None
        
    def load_models(self):
        """Load trained model(s)"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model_data = joblib.load(self.model_path)
        self.model_type = self.model_data['model_type']
        self.feature_cols = self.model_data['feature_cols']
        
        if self.model_type == 'lstm':
            lstm_path = self.model_data['lstm_path']
            self.lstm_model = load_model(lstm_path)
            print(f"Loaded LSTM model from {lstm_path}")
        else:
            self.model = self.model_data['model']
            print(f"Loaded {self.model_type} model")
    
    def fetch_latest_data(self, ticker, days=250):
        """Fetch recent stock data"""
        df = yf.download(ticker, period=f"{days}d", interval="1d")
        df.dropna(inplace=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        return df
    
    def prepare_features(self, df):
        """Engineer features for prediction"""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        return df_features
    
    def predict_single(self, features):
        """Make prediction based on model type"""
        if self.model_type == 'ensemble':
            # Ensemble prediction (average probabilities)
            predictions = []
            for name, model in self.model.items():
                pred = model.predict_proba(features)[0][1]
                predictions.append(pred)
            avg_prob = np.mean(predictions)
            prediction = 1 if avg_prob > 0.5 else 0
            confidence = avg_prob if prediction == 1 else (1 - avg_prob)
            
            # Individual model predictions
            individual_preds = {}
            for name, model in self.model.items():
                individual_preds[name] = model.predict(features)[0]
            
            return prediction, confidence, individual_preds
        
        elif self.model_type == 'lstm':
            # LSTM requires sequence
            seq_length = 10
            if len(features) < seq_length:
                raise ValueError(f"Need at least {seq_length} days of data for LSTM")
            
            X_seq = features[-seq_length:].values.reshape(1, seq_length, -1)
            prob = self.lstm_model.predict(X_seq, verbose=0)[0][0]
            prediction = 1 if prob > 0.5 else 0
            confidence = prob if prediction == 1 else (1 - prob)
            
            return prediction, confidence, None
        
        else:
            # Single model prediction
            prediction = self.model.predict(features)[0]
            if hasattr(self.model, 'predict_proba'):
                prob = self.model.predict_proba(features)[0][1]
                confidence = prob if prediction == 1 else (1 - prob)
            else:
                confidence = None
            
            return prediction, confidence, None
    
    def predict(self, ticker):
        """Full prediction pipeline"""
        print(f"Fetching data for {ticker}...")
        df = self.fetch_latest_data(ticker)
        
        print("Engineering features...")
        df_features = self.prepare_features(df)
        
        print("Loading model...")
        self.load_models()
        
        # Get latest features
        if self.model_type == 'lstm':
            # Use last 10 rows for LSTM
            latest_features = df_features[self.feature_cols][-10:]
        else:
            latest_features = df_features[self.feature_cols].iloc[-1:].values
        
        # Make prediction
        prediction, confidence, individual_preds = self.predict_single(latest_features)
        
        # Display results
        self.display_results(df_features, prediction, confidence, individual_preds)
    
    def display_results(self, df, prediction, confidence, individual_preds):
        """Display prediction results"""
        print("\n" + "="*50)
        print("STOCK PREDICTION RESULTS")
        print("="*50)
        
        last_date = df.index[-1].date()
        last_close = df['Close'].iloc[-1]
        
        print(f"\nLast trading day: {last_date}")
        print(f"Last close price: ${last_close:.2f}")
        
        if prediction == 1:
            print("\n📈 PREDICTION: UP tomorrow")
        else:
            print("\n📉 PREDICTION: DOWN tomorrow")
        
        if confidence:
            print(f"Confidence: {confidence*100:.1f}%")
        
        # Show individual model predictions for ensemble
        if individual_preds:
            print("\nIndividual Model Predictions:")
            for model_name, pred in individual_preds.items():
                direction = "UP" if pred == 1 else "DOWN"
                print(f"  - {model_name}: {direction}")
        
        # Show recent indicators
        print("\nRecent Technical Indicators:")
        if 'RSI' in df.columns:
            print(f"  RSI: {df['RSI'].iloc[-1]:.2f}")
        if 'MACD' in df.columns:
            print(f"  MACD: {df['MACD'].iloc[-1]:.2f}")
        if 'Volatility' in df.columns:
            print(f"  Volatility: {df['Volatility'].iloc[-1]:.4f}")
        
        print("\n" + "="*50)


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <TICKER>")
        print("Example: python predict.py AAPL")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    try:
        predictor = StockPredictor()
        predictor.predict(ticker)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()