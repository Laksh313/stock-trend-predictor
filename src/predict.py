import sys
import os
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
from tensorflow.keras.models import load_model

# Import sentiment analysis
try:
    from news_sentiment import get_stock_sentiment, NewsSentimentAnalyzer
    from sentiment_config import SENTIMENT_CONFIG
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("⚠️  Sentiment analysis not available. Install required packages or check configuration.")

MODEL_PATH = "model/advanced_ensemble_model.pkl"  # Default to advanced model

class StockPredictor:
    """Unified predictor for all model types including advanced models"""
    
    def __init__(self, model_path=MODEL_PATH, use_sentiment=True):
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.model_type = None
        self.feature_cols = None
        self.lstm_model = None
        self.scaler = None
        self.use_scaling = False
        self.use_sentiment = use_sentiment and SENTIMENT_AVAILABLE
        
    def create_lagged_features(self, df, feature_cols, lags=[1, 2, 3]):
        """Create lagged features for advanced models"""
        df_lagged = df.copy()
        
        for col in feature_cols:
            if col in df.columns and col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']:
                for lag in lags:
                    df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df_lagged
    
    def create_rolling_features(self, df, feature_cols, windows=[3, 5]):
        """Create rolling statistics features for advanced models"""
        df_rolling = df.copy()
        
        for col in feature_cols:
            if col in df.columns and col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']:
                for window in windows:
                    df_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
        
        return df_rolling
        
    def load_models(self):
        """Load trained model(s)"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model_data = joblib.load(self.model_path)
        self.model_type = self.model_data['model_type']
        self.feature_cols = self.model_data['feature_cols']
        
        # Check if model has scaler (advanced models)
        if 'scaler' in self.model_data and self.model_data['scaler'] is not None:
            self.scaler = self.model_data['scaler']
            self.use_scaling = self.model_data.get('use_scaling', False)
            print(f"Loaded model with scaling enabled")
        
        if self.model_type == 'lstm':
            lstm_path = self.model_data['lstm_path']
            self.lstm_model = load_model(lstm_path)
            print(f"Loaded LSTM model from {lstm_path}")
        else:
            self.model = self.model_data['model']
            print(f"Loaded {self.model_type} model")
    
    def fetch_latest_data(self, ticker, days=300):
        """Fetch recent stock data (need more days for lagged features)"""
        df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
        df.dropna(inplace=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        return df
    
    def prepare_features(self, df, is_advanced_model=False):
        """Engineer features for prediction"""
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df)
        
        # If this is an advanced model, create lagged and rolling features
        if is_advanced_model:
            print("Creating advanced features (lagged and rolling)...")
            
            # Get base feature columns (exclude OHLCV and Target)
            base_features = [col for col in df_features.columns 
                           if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Create lagged features
            df_features = self.create_lagged_features(df_features, base_features, lags=[1, 2, 3])
            
            # Create rolling features
            df_features = self.create_rolling_features(df_features, base_features, windows=[3, 5])
            
            # Drop NaN values created by lagging and rolling
            df_features.dropna(inplace=True)
        
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
        
        print("Loading model...")
        self.load_models()
        
        # Check if this is an advanced model by looking at feature names
        is_advanced = any('_lag_' in col or '_rolling_' in col for col in self.feature_cols)
        
        if is_advanced:
            print("Detected advanced model (with lagged/rolling features)")
        
        print("Engineering features...")
        df_features = self.prepare_features(df, is_advanced_model=is_advanced)
        
        # Check if we have all required features
        missing_features = set(self.feature_cols) - set(df_features.columns)
        if missing_features:
            print(f"\n⚠️  Warning: Missing {len(missing_features)} features")
            print(f"This might happen if the model was trained with different feature settings")
            raise ValueError(f"Missing required features. Please retrain the model or use a compatible model.")
        
        # Select only the features the model needs, in the correct order
        features_df = df_features[self.feature_cols]
        
        # Apply scaling if the model was trained with scaling
        if self.use_scaling and self.scaler is not None:
            print("Applying feature scaling...")
            features_df = pd.DataFrame(
                self.scaler.transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
        
        # Get latest features
        if self.model_type == 'lstm':
            # Use last 10 rows for LSTM
            latest_features = features_df[-10:]
        else:
            latest_features = features_df.iloc[-1:].values
        
        # Make prediction
        prediction, confidence, individual_preds = self.predict_single(latest_features)
        
        # Get sentiment analysis (NEW)
        sentiment_data = None
        if self.use_sentiment:
            sentiment_data = get_stock_sentiment(ticker)
        
        # Display results with sentiment
        self.display_results(df_features, prediction, confidence, individual_preds, sentiment_data)
    
    def display_results(self, df, prediction, confidence, individual_preds, sentiment_data=None):
        """Display prediction results with sentiment analysis"""
        print("\n" + "="*70)
        print("STOCK PREDICTION RESULTS")
        print("="*70)
        
        # Basic info
        ticker_name = df.index.name if hasattr(df.index, 'name') else "Stock"
        last_date = df.index[-1].date()
        last_close = df['Close'].iloc[-1]
        
        print(f"\nLast trading day: {last_date}")
        print(f"Last close price: ${last_close:.2f}")
        
        # Model prediction section
        print(f"\n{'='*70}")
        print("MODEL PREDICTION:")
        print(f"{'='*70}")
        
        if prediction == 1:
            print("📈 PREDICTION: UP tomorrow")
        else:
            print("📉 PREDICTION: DOWN tomorrow")
        
        if confidence:
            print(f"Confidence: {confidence*100:.1f}%")
        
        # Show individual model predictions for ensemble
        if individual_preds:
            print("\nIndividual Model Predictions:")
            for model_name, pred in individual_preds.items():
                direction = "UP" if pred == 1 else "DOWN"
                print(f"  - {model_name}: {direction}")
        
        # Sentiment analysis section (NEW)
        if sentiment_data and sentiment_data.get('available'):
            print(f"\n{'='*70}")
            print("NEWS SENTIMENT ANALYSIS:")
            print(f"{'='*70}")
            
            print(f"\n{sentiment_data['sentiment_icon']} Overall Sentiment: {sentiment_data['sentiment_label'].upper()}")
            print(f"   Sentiment Score: {sentiment_data['overall_sentiment']:+.2f} (range: -1 to +1)")
            print(f"📊 Sentiment Trend: {sentiment_data['sentiment_trend']}")
            print(f"📑 Articles Analyzed: {sentiment_data['news_count']} (last {SENTIMENT_CONFIG['news_days']} days)")
            
            # Combined analysis (NEW)
            print(f"\n{'='*70}")
            print("COMBINED ANALYSIS:")
            print(f"{'='*70}")
            
            if sentiment_data['available'] and confidence:
                # Analyze agreement
                analyzer = NewsSentimentAnalyzer()
                agreement = analyzer.analyze_agreement(
                    prediction, 
                    confidence, 
                    sentiment_data['overall_sentiment']
                )
                
                print(f"\n{agreement['message']}")
                print(f"Overall Confidence: {agreement['confidence'].upper()}")
                
                # Detailed breakdown
                model_dir = "UP" if prediction == 1 else "DOWN"
                sent_dir = "POSITIVE" if sentiment_data['overall_sentiment'] > 0.1 else \
                          "NEGATIVE" if sentiment_data['overall_sentiment'] < -0.1 else "NEUTRAL"
                
                print(f"\nBreakdown:")
                print(f"  • Technical Indicators: {model_dir} ({confidence*100:.1f}% confidence)")
                print(f"  • News Sentiment: {sent_dir} ({abs(sentiment_data['overall_sentiment']):.2f} strength)")
                
                # Trading suggestion
                if agreement['type'] == 'strong_agreement':
                    print(f"\n💡 Suggestion: This is a high-confidence signal. Both technical and fundamental indicators align.")
                elif agreement['type'] == 'weak_agreement':
                    print(f"\n💡 Suggestion: Moderate confidence. Indicators agree but signals are not very strong.")
                elif agreement['type'] == 'neutral_sentiment':
                    print(f"\n💡 Suggestion: Rely primarily on technical analysis. News doesn't provide clear direction.")
                else:  # disagreement
                    print(f"\n💡 Suggestion: Exercise caution. Technical and news sentiment are contradicting.")
                    print(f"   Consider waiting for more clarity or using smaller position sizes.")
        
        elif sentiment_data and not sentiment_data.get('available'):
            print(f"\n{'='*70}")
            print("NEWS SENTIMENT ANALYSIS:")
            print(f"{'='*70}")
            print(f"\n⚠️  {sentiment_data.get('message', 'Sentiment analysis unavailable')}")
            print("   Relying solely on technical indicators for this prediction.")
        
        # Technical indicators section
        print(f"\n{'='*70}")
        print("RECENT TECHNICAL INDICATORS:")
        print(f"{'='*70}")
        
        if 'RSI' in df.columns:
            rsi_val = df['RSI'].iloc[-1]
            rsi_signal = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
            print(f"  RSI: {rsi_val:.2f} ({rsi_signal})")
        
        if 'MACD' in df.columns:
            macd_val = df['MACD'].iloc[-1]
            macd_signal = "Bullish" if macd_val > 0 else "Bearish"
            print(f"  MACD: {macd_val:.2f} ({macd_signal})")
        
        if 'Volatility' in df.columns:
            vol_val = df['Volatility'].iloc[-1]
            vol_signal = "High" if vol_val > 0.03 else "Low" if vol_val < 0.01 else "Normal"
            print(f"  Volatility: {vol_val:.4f} ({vol_signal})")
        
        if 'BB_Width' in df.columns:
            print(f"  Bollinger Band Width: {df['BB_Width'].iloc[-1]:.4f}")
        
        print("\n" + "="*70)
        print("⚠️  DISCLAIMER: Not financial advice. For educational purposes only.")
        print("="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <TICKER> [MODEL_PATH] [OPTIONS]")
        print("\nExamples:")
        print("  python predict.py AAPL")
        print("  python predict.py AAPL model/advanced_ensemble_model.pkl")
        print("  python predict.py TSLA --no-sentiment")
        print("\nOptions:")
        print("  --no-sentiment    Disable news sentiment analysis")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    # Parse arguments
    model_path = MODEL_PATH
    use_sentiment = True
    
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == '--no-sentiment':
            use_sentiment = False
        elif not arg.startswith('--'):
            model_path = arg
    
    try:
        predictor = StockPredictor(model_path, use_sentiment=use_sentiment)
        predictor.predict(ticker)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()