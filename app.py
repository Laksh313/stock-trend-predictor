"""
Flask Web Application for Stock Trend Predictor
"""

from flask import Flask, render_template, request, jsonify
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict import StockPredictor
from news_sentiment import get_stock_sentiment, NewsSentimentAnalyzer
from sentiment_config import SENTIMENT_CONFIG
import traceback
import pandas as pd

app = Flask(__name__)

# Configuration
MODEL_PATH = "model/advanced_ensemble_model.pkl"  # Default model

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request"""
    try:
        data = request.get_json()
        ticker = data.get('ticker', '').upper().strip()
        use_sentiment = data.get('use_sentiment', True)
        
        if not ticker:
            return jsonify({
                'success': False,
                'error': 'Please enter a stock ticker symbol'
            })
        
        # Validate ticker (basic check)
        if len(ticker) > 5 or not ticker.isalpha():
            return jsonify({
                'success': False,
                'error': 'Invalid ticker symbol'
            })
        
        # Create predictor
        predictor = StockPredictor(MODEL_PATH, use_sentiment=use_sentiment)
        
        # Fetch data
        df = predictor.fetch_latest_data(ticker)
        if df is None or len(df) == 0:
            return jsonify({
                'success': False,
                'error': f'No data found for {ticker}. Please check the ticker symbol.'
            })
        
        # Load model
        predictor.load_models()
        
        # Check if advanced model
        is_advanced = any('_lag_' in col or '_rolling_' in col for col in predictor.feature_cols)
        
        # Prepare features
        df_features = predictor.prepare_features(df, is_advanced_model=is_advanced)
        
        # Check for missing features
        missing_features = set(predictor.feature_cols) - set(df_features.columns)
        if missing_features:
            return jsonify({
                'success': False,
                'error': 'Model feature mismatch. Please retrain the model.'
            })
        
        # Select and scale features
        features_df = df_features[predictor.feature_cols]
        
        if predictor.use_scaling and predictor.scaler is not None:
            features_df = pd.DataFrame(
                predictor.scaler.transform(features_df),
                columns=features_df.columns,
                index=features_df.index
            )
        
        # Get latest features
        if predictor.model_type == 'lstm':
            latest_features = features_df[-10:]
        else:
            latest_features = features_df.iloc[-1:].values
        
        # Make prediction
        prediction, confidence, individual_preds = predictor.predict_single(latest_features)
        
        # Get sentiment if enabled
        sentiment_data = None
        if use_sentiment:
            sentiment_data = get_stock_sentiment(ticker)
        
        # Prepare response
        last_date = df_features.index[-1].strftime('%Y-%m-%d')
        last_close = float(df_features['Close'].iloc[-1])
        
        # Technical indicators
        technical_indicators = {}
        if 'RSI' in df_features.columns:
            rsi_val = float(df_features['RSI'].iloc[-1])
            technical_indicators['RSI'] = {
                'value': round(rsi_val, 2),
                'signal': 'Overbought' if rsi_val > 70 else 'Oversold' if rsi_val < 30 else 'Neutral'
            }
        
        if 'MACD' in df_features.columns:
            macd_val = float(df_features['MACD'].iloc[-1])
            technical_indicators['MACD'] = {
                'value': round(macd_val, 2),
                'signal': 'Bullish' if macd_val > 0 else 'Bearish'
            }
        
        if 'Volatility' in df_features.columns:
            vol_val = float(df_features['Volatility'].iloc[-1])
            technical_indicators['Volatility'] = {
                'value': round(vol_val, 4),
                'signal': 'High' if vol_val > 0.03 else 'Low' if vol_val < 0.01 else 'Normal'
            }
        
        # Build response
        response = {
            'success': True,
            'ticker': ticker,
            'last_date': last_date,
            'last_close': round(last_close, 2),
            'prediction': {
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': round(confidence * 100, 1) if confidence else None,
                'model_type': predictor.model_type
            },
            'technical_indicators': technical_indicators
        }
        
        # Add individual predictions for ensemble
        if individual_preds:
            response['prediction']['individual_models'] = {
                name: 'UP' if pred == 1 else 'DOWN' 
                for name, pred in individual_preds.items()
            }
        
        # Add sentiment data
        if sentiment_data and sentiment_data.get('available'):
            response['sentiment'] = {
                'available': True,
                'score': round(sentiment_data['overall_sentiment'], 2),
                'label': sentiment_data['sentiment_label'],
                'icon': sentiment_data['sentiment_icon'],
                'trend': sentiment_data['sentiment_trend'],
                'articles_count': sentiment_data['news_count']
            }
            
            # Calculate agreement
            if confidence:
                analyzer = NewsSentimentAnalyzer()
                agreement = analyzer.analyze_agreement(
                    prediction, 
                    confidence, 
                    sentiment_data['overall_sentiment']
                )
                response['sentiment']['agreement'] = {
                    'type': agreement['type'],
                    'message': agreement['message'],
                    'confidence': agreement['confidence']
                }
        elif sentiment_data:
            response['sentiment'] = {
                'available': False,
                'message': sentiment_data.get('message', 'No sentiment data available')
            }
        
        return jsonify(response)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {error_trace}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Warning: Model not found at {MODEL_PATH}")
        print("Please train a model first using train_advanced.py")
    
    print("🚀 Starting Stock Trend Predictor Web App...")
    print("📊 Access at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)