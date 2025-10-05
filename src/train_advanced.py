"""
Advanced training with feature selection, data augmentation, and optimization
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from data_fetch import fetch_stock_data
from data_preprocessing import load_and_clean
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
import joblib
import os


def create_lagged_features(df, feature_cols, lags=[1, 2, 3, 5]):
    """Create lagged features for better predictions"""
    df_lagged = df.copy()
    
    for col in feature_cols:
        if col in df.columns and col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']:
            for lag in lags:
                df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df_lagged


def create_rolling_features(df, feature_cols, windows=[3, 5, 10]):
    """Create rolling statistics features"""
    df_rolling = df.copy()
    
    for col in feature_cols:
        if col in df.columns and col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume']:
            for window in windows:
                df_rolling[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                df_rolling[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
    
    return df_rolling


def augment_data(df, augmentation_factor=2):
    """Augment data by adding noise to minority class"""
    target_col = 'Target'
    
    # Separate classes
    df_majority = df[df[target_col] == df[target_col].value_counts().idxmax()]
    df_minority = df[df[target_col] == df[target_col].value_counts().idxmin()]
    
    print(f"\nOriginal class distribution:")
    print(f"  Majority: {len(df_majority)}")
    print(f"  Minority: {len(df_minority)}")
    
    # Generate synthetic samples with noise
    augmented_samples = []
    feature_cols = [col for col in df.columns if col != target_col]
    
    for _ in range(augmentation_factor):
        noisy_minority = df_minority.copy()
        # Add small random noise (1-3% of std)
        for col in feature_cols:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                noise = np.random.normal(0, df[col].std() * 0.02, len(noisy_minority))
                noisy_minority[col] = noisy_minority[col] + noise
        augmented_samples.append(noisy_minority)
    
    # Combine all data
    df_augmented = pd.concat([df_majority, df_minority] + augmented_samples, ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"After augmentation:")
    print(f"  Total samples: {len(df_augmented)}")
    print(f"  Class 0: {len(df_augmented[df_augmented[target_col]==0])}")
    print(f"  Class 1: {len(df_augmented[df_augmented[target_col]==1])}")
    
    return df_augmented


def select_best_features(X_train, y_train, X_test, max_features=30):
    """Select best features using LightGBM feature importance"""
    print(f"\nSelecting top {max_features} features...")
    
    # Train a quick model for feature selection
    selector = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    selector.fit(X_train, y_train)
    
    # Get feature importance
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': selector.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 most important features:")
    for idx, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Select top features
    top_features = importance.head(max_features)['feature'].tolist()
    
    return X_train[top_features], X_test[top_features], top_features


def scale_features(X_train, X_test):
    """Scale features to improve model performance"""
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler


def cross_validate_model(X, y, model_type='lightgbm', n_splits=5):
    """Perform time series cross-validation"""
    print(f"\nPerforming {n_splits}-fold time series cross-validation...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_cv = X.iloc[train_idx]
        y_train_cv = y.iloc[train_idx]
        X_val_cv = X.iloc[val_idx]
        y_val_cv = y.iloc[val_idx]
        
        # Train model
        trainer = ModelTrainer(model_type=model_type)
        
        if model_type == 'lightgbm':
            model = trainer.train_lightgbm(X_train_cv, y_train_cv)
        elif model_type == 'random_forest':
            model = trainer.train_random_forest(X_train_cv, y_train_cv)
        else:
            model = trainer.train_stacker(X_train_cv, y_train_cv)
        
        # Evaluate
        y_pred = model.predict(X_val_cv)
        score = accuracy_score(y_val_cv, y_pred)
        scores.append(score)
        
        print(f"  Fold {fold}: {score:.4f}")
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"\nCV Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
    
    return mean_score, std_score


def train_advanced_model(tickers, model_type='lightgbm', start_date='2019-01-01', 
                        end_date='2024-12-31', use_augmentation=True, 
                        use_feature_selection=True, use_scaling=True,
                        max_features=30):
    """Advanced training pipeline"""
    
    print("="*70)
    print("ADVANCED MODEL TRAINING")
    print("="*70)
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Model: {model_type}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Augmentation: {use_augmentation}")
    print(f"Feature Selection: {use_feature_selection}")
    print(f"Scaling: {use_scaling}")
    print("="*70)
    
    # Step 1: Fetch and combine data
    print("\n[1/7] Fetching stock data...")
    all_data = []
    for ticker in tickers:
        try:
            fetch_stock_data(ticker, start_date, end_date)
            df = load_and_clean(f"data/{ticker}_data.csv")
            df['Ticker'] = ticker
            all_data.append(df)
            print(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
    
    if not all_data:
        raise ValueError("No data fetched!")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} rows")
    
    # Step 2: Feature engineering
    print("\n[2/7] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(combined_df)
    
    # Get base feature columns
    base_features = [col for col in df_features.columns 
                    if col not in ['Target', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Step 3: Create advanced features
    print("\n[3/7] Creating lagged and rolling features...")
    df_features = create_lagged_features(df_features, base_features, lags=[1, 2, 3])
    df_features = create_rolling_features(df_features, base_features, windows=[3, 5])
    df_features.dropna(inplace=True)
    
    if 'Ticker' in df_features.columns:
        df_features = df_features.drop('Ticker', axis=1)
    
    print(f"Total features after engineering: {len(df_features.columns) - 1}")
    
    # Step 4: Data augmentation
    if use_augmentation:
        print("\n[4/7] Augmenting data...")
        df_features = augment_data(df_features, augmentation_factor=1)
    else:
        print("\n[4/7] Skipping augmentation")
    
    # Step 5: Prepare train/test split
    print("\n[5/7] Preparing train/test split...")
    feature_cols = [col for col in df_features.columns if col != 'Target']
    X = df_features[feature_cols]
    y = df_features['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 6: Feature selection
    if use_feature_selection:
        print(f"\n[6/7] Selecting best {max_features} features...")
        X_train, X_test, selected_features = select_best_features(
            X_train, y_train, X_test, max_features
        )
    else:
        print("\n[6/7] Skipping feature selection")
        selected_features = feature_cols
    
    # Step 7: Scaling
    scaler = None
    if use_scaling:
        print("\n[7/7] Scaling features...")
        X_train, X_test, scaler = scale_features(X_train, X_test)
    else:
        print("\n[7/7] Skipping scaling")
    
    # Cross-validation
    print("\n" + "="*70)
    cross_validate_model(X_train, y_train, model_type=model_type, n_splits=5)
    
    # Final training
    print("\n" + "="*70)
    print("Training final model...")
    trainer = ModelTrainer(model_type=model_type)
    
    if model_type == 'lightgbm':
        model = trainer.train_lightgbm(X_train, y_train)
    elif model_type == 'random_forest':
        model = trainer.train_random_forest(X_train, y_train)
    elif model_type == 'stacker':
        model = trainer.train_stacker(X_train, y_train)
    elif model_type == 'ensemble':
        model = trainer.train_ensemble(X_train, y_train, X_test, y_test)
        # For ensemble, predict differently
        predictions = []
        for name, m in model.items():
            predictions.append(m.predict(X_test))
        y_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if model_type != 'ensemble':
        y_pred = model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Final Test Accuracy: {acc:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    # Save model
    output_path = f"model/advanced_{model_type}_model.pkl"
    os.makedirs("model", exist_ok=True)
    
    model_data = {
        'model': model,
        'model_type': model_type,
        'feature_cols': selected_features,
        'scaler': scaler,
        'use_scaling': use_scaling
    }
    
    joblib.dump(model_data, output_path)
    print(f"\nModel saved to: {output_path}")
    
    return acc


def main():
    parser = argparse.ArgumentParser(description='Advanced model training with optimization')
    
    parser.add_argument('--tickers', nargs='+', required=True, help='Stock tickers')
    parser.add_argument('--model', default='lightgbm', 
                       choices=['lightgbm', 'random_forest', 'stacker', 'ensemble'])
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--no-augmentation', action='store_true')
    parser.add_argument('--no-feature-selection', action='store_true')
    parser.add_argument('--no-scaling', action='store_true')
    parser.add_argument('--max-features', type=int, default=30)
    
    args = parser.parse_args()
    
    train_advanced_model(
        tickers=[t.upper() for t in args.tickers],
        model_type=args.model,
        start_date=args.start,
        end_date=args.end,
        use_augmentation=not args.no_augmentation,
        use_feature_selection=not args.no_feature_selection,
        use_scaling=not args.no_scaling,
        max_features=args.max_features
    )


if __name__ == '__main__':
    main()