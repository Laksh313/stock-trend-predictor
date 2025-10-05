import pandas as pd
import ta
import numpy as np

class FeatureEngineer:
    """Modular feature engineering for stock data"""
    
    def __init__(self):
        self.feature_groups = {
            'basic': self._add_basic_features,
            'momentum': self._add_momentum_features,
            'volatility': self._add_volatility_features,
            'volume': self._add_volume_features,
            'patterns': self._add_pattern_features,
            'advanced': self._add_advanced_features
        }
    
    def _flatten_columns(self, df):
        """Flatten multi-index columns from yfinance"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        return df
    
    def _ensure_numeric(self, df, cols):
        """Ensure columns are numeric"""
        for col in cols:
            if col in df.columns:
                if isinstance(df[col], pd.DataFrame) or df[col].ndim > 1:
                    df[col] = df[col].iloc[:, 0]
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    
    def _add_basic_features(self, df):
        """Add basic return and moving average features"""
        df["Return"] = df["Close"].pct_change()
        df["SMA_10"] = df["Close"].rolling(10).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["SMA_200"] = df["Close"].rolling(200).mean()
        df["EMA_10"] = df["Close"].ewm(span=10).mean()
        df["EMA_20"] = df["Close"].ewm(span=20).mean()
        return df
    
    def _add_momentum_features(self, df):
        """Add momentum indicators"""
        close = pd.Series(df["Close"].values, index=df.index)
        
        # RSI
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        df["RSI_SMA"] = df["RSI"].rolling(10).mean()
        
        # MACD
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Diff"] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(
            df["High"], df["Low"], df["Close"]
        )
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()
        
        # Williams %R
        df["Williams_R"] = ta.momentum.WilliamsRIndicator(
            df["High"], df["Low"], df["Close"]
        ).williams_r()
        
        return df
    
    def _add_volatility_features(self, df):
        """Add volatility indicators"""
        df["Volatility"] = df["Return"].rolling(10).std()
        df["Volatility_20"] = df["Return"].rolling(20).std()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df["Close"])
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()
        df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / df["BB_Mid"]
        
        # Average True Range
        df["ATR"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range()
        
        return df
    
    def _add_volume_features(self, df):
        """Add volume-based features"""
        df["Volume_SMA"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
        
        # On-Balance Volume
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            df["Close"], df["Volume"]
        ).on_balance_volume()
        
        # Money Flow Index
        df["MFI"] = ta.volume.MFIIndicator(
            df["High"], df["Low"], df["Close"], df["Volume"]
        ).money_flow_index()
        
        return df
    
    def _add_pattern_features(self, df):
        """Add price pattern features"""
        # High-Low range
        df["HL_Range"] = df["High"] - df["Low"]
        df["HL_Range_Pct"] = df["HL_Range"] / df["Close"]
        
        # Gap detection
        df["Gap"] = df["Open"] - df["Close"].shift(1)
        df["Gap_Pct"] = df["Gap"] / df["Close"].shift(1)
        
        # Upper/Lower shadow (candlestick)
        df["Upper_Shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["Lower_Shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        
        # Body size
        df["Body_Size"] = abs(df["Close"] - df["Open"])
        df["Body_Size_Pct"] = df["Body_Size"] / df["Close"]
        
        return df
    
    def _add_advanced_features(self, df):
        """Add advanced derived features"""
        # Price momentum
        df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
        df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
        
        # Rate of Change
        df["ROC_5"] = df["Close"].pct_change(5)
        df["ROC_10"] = df["Close"].pct_change(10)
        
        # Distance from moving averages
        df["Dist_SMA_10"] = (df["Close"] - df["SMA_10"]) / df["SMA_10"]
        df["Dist_SMA_50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]
        
        # Trend strength
        df["Trend_Strength"] = df["SMA_10"] - df["SMA_50"]
        
        # Historical High/Low
        df["High_52W"] = df["High"].rolling(252).max()
        df["Low_52W"] = df["Low"].rolling(252).min()
        df["Dist_High_52W"] = (df["Close"] - df["High_52W"]) / df["High_52W"]
        df["Dist_Low_52W"] = (df["Close"] - df["Low_52W"]) / df["Low_52W"]
        
        return df
    
    def engineer_features(self, df, feature_groups=None):
        """
        Apply feature engineering to dataframe
        
        Args:
            df: Input dataframe with OHLCV data
            feature_groups: List of feature groups to apply. 
                          If None, applies all groups.
                          Options: 'basic', 'momentum', 'volatility', 'volume', 
                                   'patterns', 'advanced'
        
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        df = self._flatten_columns(df)
        df = self._ensure_numeric(df, ["Open", "High", "Low", "Close", "Volume"])
        
        # Apply selected feature groups
        if feature_groups is None:
            feature_groups = list(self.feature_groups.keys())
        
        for group in feature_groups:
            if group in self.feature_groups:
                print(f"Adding {group} features...")
                df = self.feature_groups[group](df)
            else:
                print(f"Warning: Unknown feature group '{group}'")
        
        # Add target (must be last)
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        
        # Clean up
        df.dropna(inplace=True)
        
        return df
    
    def get_feature_list(self, feature_groups=None):
        """
        Get list of feature column names for given feature groups
        
        Args:
            feature_groups: List of feature groups
        
        Returns:
            List of feature column names
        """
        # Create dummy dataframe to extract feature names
        dummy_df = pd.DataFrame({
            'Open': [100] * 300,
            'High': [101] * 300,
            'Low': [99] * 300,
            'Close': [100] * 300,
            'Volume': [1000000] * 300
        })
        
        engineered_df = self.engineer_features(dummy_df, feature_groups)
        
        # Exclude OHLCV and Target
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
        feature_cols = [col for col in engineered_df.columns if col not in exclude_cols]
        
        return feature_cols


# Convenience function for backward compatibility
def engineer_features(df, feature_groups=None):
    """
    Engineer features using the FeatureEngineer class
    
    Args:
        df: Input dataframe
        feature_groups: List of feature groups to apply (default: all)
    
    Returns:
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(df, feature_groups)