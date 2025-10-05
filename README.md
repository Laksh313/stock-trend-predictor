# Stock Trend Predictor

A flexible machine learning framework for predicting stock price movements using multiple models and customizable features.

## Features

- **Multiple Models**: Random Forest, LightGBM, LSTM, Stacking Ensemble, and Combined Ensemble
- **Modular Feature Engineering**: Easily add or remove feature groups
- **Extensible Architecture**: Simple to add new features and models
- **Comprehensive Technical Indicators**: 40+ features including momentum, volatility, volume, and pattern recognition

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train a Model

```bash
# Train ensemble model (recommended)
python src/main.py AAPL --model ensemble

# Train specific model
python src/main.py AAPL --model lightgbm

# Train with specific features
python src/main.py AAPL --model random_forest --features basic momentum volatility

# Custom date range
python src/main.py AAPL --model lstm --start 2019-01-01 --end 2024-12-31
```

### 2. Make Predictions

```bash
python src/predict.py AAPL
```

## Available Models

| Model           | Description                      | Best For                          |
| --------------- | -------------------------------- | --------------------------------- |
| `random_forest` | Random Forest Classifier         | Stable, interpretable predictions |
| `lightgbm`      | LightGBM Gradient Boosting       | Fast training, high accuracy      |
| `lstm`          | LSTM Neural Network              | Sequential pattern recognition    |
| `stacker`       | Stacked Ensemble (RF + LightGBM) | Combining model strengths         |
| `ensemble`      | All models combined              | Best overall accuracy             |

## Feature Groups

| Group        | Description         | Features Included                        |
| ------------ | ------------------- | ---------------------------------------- |
| `basic`      | Core features       | Returns, SMA, EMA                        |
| `momentum`   | Momentum indicators | RSI, MACD, Stochastic, Williams %R       |
| `volatility` | Volatility measures | Standard deviation, Bollinger Bands, ATR |
| `volume`     | Volume-based        | Volume ratios, OBV, MFI                  |
| `patterns`   | Price patterns      | Candlestick patterns, gaps, shadows      |
| `advanced`   | Derived features    | ROC, momentum, distance from MAs         |

## Project Structure

```
stock-trend-predictor/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Main training script
│   ├── predict.py                 # Prediction script
│   ├── data_fetch.py             # Data fetching from yfinance
│   ├── data_preprocessing.py     # Data cleaning
│   ├── feature_engineering.py    # Feature engineering module
│   ├── model_training.py         # Model training module
│   └── config.py                 # Configuration file
├── data/                          # Downloaded stock data
├── model/                         # Trained models
├── requirements.txt
└── README.md
```

## Adding Custom Features

To add new features, edit `src/feature_engineering.py`:

```python
def _add_custom_features(self, df):
    """Add your custom features here"""
    # Example: Add custom indicator
    df["My_Indicator"] = df["Close"].rolling(15).mean() / df["Close"]
    return df

# Register in __init__
self.feature_groups['custom'] = self._add_custom_features
```

Then train with your new features:

```bash
python src/main.py AAPL --features basic momentum custom
```

## Adding Custom Models

To add a new model, edit `src/model_training.py`:

```python
def train_my_model(self, X_train, y_train):
    """Train your custom model"""
    from some_library import MyModel
    model = MyModel(param1=value1)
    model.fit(X_train, y_train)
    return model

# Add to train() method
elif self.model_type == 'my_model':
    self.model = self.train_my_model(X_train, y_train)
    y_pred = self.model.predict(X_test)
```

## Configuration

Customize model parameters and features in `src/config.py`:

```python
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 200,  # Adjust hyperparameters
        'max_depth': 15,
        ...
    }
}

FEATURE_CONFIG = {
    'sma_periods': [10, 20, 50],  # Add/remove periods
    'rsi_window': 14,
    ...
}
```

## Examples

### Train Multiple Models

```bash
# Train and compare different models
python src/main.py AAPL --model random_forest
python src/main.py AAPL --model lightgbm
python src/main.py AAPL --model lstm
python src/main.py AAPL --model ensemble
```

### Different Tickers

```bash
python src/main.py TSLA --model ensemble
python src/main.py MSFT --model lightgbm --features basic momentum
python src/main.py GOOGL --model stacker
```

### Minimal Features (Faster Training)

```bash
python src/main.py AAPL --model random_forest --features basic momentum
```

### All Features (Maximum Information)

```bash
python src/main.py AAPL --model ensemble --features basic momentum volatility volume patterns advanced
```

## Model Performance

The model outputs:

- **Test Accuracy**: Overall prediction accuracy
- **Classification Report**: Precision, recall, F1-score for UP/DOWN predictions
- **Confidence Score**: Prediction confidence (for ensemble models)
- **Individual Predictions**: Each model's prediction in ensemble mode

## Tips for Better Accuracy

1. **Use more data**: Longer date ranges provide better training
2. **Feature selection**: Experiment with different feature groups
3. **Ensemble models**: Usually provide best results
4. **Regular retraining**: Retrain models with recent data
5. **Multiple stocks**: Some models work better for certain stocks

## Limitations

- Predictions are binary (UP/DOWN), not price targets
- Past performance doesn't guarantee future results
- Market conditions change; models need regular updates
- Not financial advice - use for educational purposes only

## Contributing

Feel free to add:

- New technical indicators
- Additional models (XGBoost, CatBoost, etc.)
- Alternative prediction targets (price ranges, volatility)
- Backtesting functionality
- Model interpretation tools

## License

MIT License - feel free to use and modify
