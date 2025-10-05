"""
Configuration file for stock prediction models
Customize model parameters and feature engineering here
"""

# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 300,  # Increased for multi-stock
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'  # Handle imbalanced data
    },
    
    'lightgbm': {
        'n_estimators': 300,
        'learning_rate': 0.03,
        'max_depth': 15,
        'num_leaves': 50,
        'min_child_samples': 20,
        'random_state': 42,
        'verbose': -1,
        'class_weight': 'balanced'
    },
    
    'lstm': {
        'sequence_length': 10,
        'lstm_units': [64, 32],  # Increased capacity
        'dropout_rate': 0.3,
        'dense_units': 32,
        'epochs': 100,
        'batch_size': 64,
        'patience': 15
    }
}

# Feature engineering configuration
FEATURE_CONFIG = {
    # Which feature groups to use by default
    'default_groups': ['basic', 'momentum', 'volatility', 'volume'],
    
    # Moving average periods
    'sma_periods': [10, 20, 50, 200],
    'ema_periods': [10, 20],
    
    # Indicator windows
    'rsi_window': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_window': 20,
    'bb_std': 2,
    'atr_window': 14,
    
    # Volatility windows
    'volatility_windows': [10, 20, 30],
    
    # Momentum periods
    'momentum_periods': [5, 10, 20],
    'roc_periods': [5, 10, 20]
}

# Data fetching configuration
DATA_CONFIG = {
    'default_start': '2020-01-01',
    'default_end': '2024-12-31',
    'prediction_lookback_days': 250,
    'min_data_points': 300
}

# Training configuration
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'shuffle': False  # Keep False for time series
}

# Paths
PATHS = {
    'data_dir': 'data',
    'model_dir': 'model',
    'results_dir': 'results',
    'single_stock_model': 'model/stock_predictor.pkl',
    'multi_stock_model': 'model/multi_stock_predictor.pkl'
}

# Feature importance threshold
FEATURE_IMPORTANCE_THRESHOLD = 0.01

# Ensemble weights (optional, for weighted voting)
ENSEMBLE_WEIGHTS = {
    'random_forest': 0.33,
    'lightgbm': 0.34,
    'stacker': 0.33
}

# Stock lists for multi-stock training
STOCK_PRESETS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD'],
    'sp500_sample': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD',
                     'DIS', 'NFLX', 'NVDA', 'BAC', 'XOM', 'WMT', 'PFE', 'KO', 'CSCO', 'INTC'],
    'diverse': ['AAPL', 'JPM', 'XOM', 'JNJ', 'WMT', 'DIS', 'BA', 'CAT', 'GE', 'F'],
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B'],
}