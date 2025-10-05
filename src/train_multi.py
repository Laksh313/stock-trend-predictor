import argparse
import pandas as pd
import os
from data_fetch import fetch_stock_data
from data_preprocessing import load_and_clean
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

# Popular stock lists for training
STOCK_LISTS = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD'],
    'sp500_sample': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'HD',
                     'DIS', 'NFLX', 'NVDA', 'BAC', 'XOM', 'WMT', 'PFE', 'KO', 'CSCO', 'INTC'],
    'diverse': ['AAPL', 'JPM', 'XOM', 'JNJ', 'WMT', 'DIS', 'BA', 'CAT', 'GE', 'F'],
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B'],
}


def fetch_multiple_stocks(tickers, start_date, end_date):
    """Fetch data for multiple stocks"""
    print(f"\nFetching data for {len(tickers)} stocks...")
    
    failed = []
    successful = []
    
    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"[{i}/{len(tickers)}] Fetching {ticker}...", end=' ')
            fetch_stock_data(ticker, start_date, end_date)
            successful.append(ticker)
            print("✓")
        except Exception as e:
            print(f"✗ Failed: {e}")
            failed.append(ticker)
    
    print(f"\nSuccessfully fetched: {len(successful)}/{len(tickers)}")
    if failed:
        print(f"Failed tickers: {', '.join(failed)}")
    
    return successful, failed


def load_and_engineer_stock(ticker, feature_groups=None):
    """Load and engineer features for a single stock"""
    try:
        data_path = f"data/{ticker}_data.csv"
        df = load_and_clean(data_path)
        
        # Add ticker column for identification
        df['Ticker'] = ticker
        
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df, feature_groups)
        
        return df_features
    except Exception as e:
        print(f"  Error processing {ticker}: {e}")
        return None


def combine_stock_data(tickers, feature_groups=None):
    """Load and combine data from multiple stocks"""
    print(f"\nProcessing features for {len(tickers)} stocks...")
    
    all_data = []
    successful = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Processing {ticker}...", end=' ')
        df = load_and_engineer_stock(ticker, feature_groups)
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            successful.append(ticker)
            print(f"✓ ({len(df)} rows)")
        else:
            print("✗ Skipped")
    
    if not all_data:
        raise ValueError("No valid stock data found!")
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✓ Combined data from {len(successful)} stocks")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Stocks: {', '.join(successful)}")
    
    return combined_df, successful


def balance_dataset(df, target_col='Target'):
    """Balance the dataset by undersampling majority class"""
    from sklearn.utils import resample
    
    # Separate majority and minority classes
    df_majority = df[df[target_col] == df[target_col].value_counts().idxmax()]
    df_minority = df[df[target_col] == df[target_col].value_counts().idxmin()]
    
    print(f"\nClass distribution before balancing:")
    print(f"  UP (1): {len(df[df[target_col]==1])}")
    print(f"  DOWN (0): {len(df[df[target_col]==0])}")
    
    # Downsample majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )
    
    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Class distribution after balancing:")
    print(f"  UP (1): {len(df_balanced[df_balanced[target_col]==1])}")
    print(f"  DOWN (0): {len(df_balanced[df_balanced[target_col]==0])}")
    
    return df_balanced


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train stock prediction models on multiple stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on tech stocks
  python train_multi.py --preset tech --model ensemble
  
  # Train on custom list
  python train_multi.py --tickers AAPL MSFT GOOGL TSLA --model lightgbm
  
  # Train on S&P 500 sample with balancing
  python train_multi.py --preset sp500_sample --model stacker --balance
  
  # Train with specific features
  python train_multi.py --preset diverse --model ensemble --features basic momentum volatility
        """
    )
    
    # Stock selection
    stock_group = parser.add_mutually_exclusive_group(required=True)
    stock_group.add_argument(
        '--tickers', '-t',
        nargs='+',
        help='List of stock tickers to train on'
    )
    stock_group.add_argument(
        '--preset', '-p',
        choices=list(STOCK_LISTS.keys()),
        help='Use a preset list of stocks'
    )
    
    # Model configuration
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['random_forest', 'lightgbm', 'lstm', 'stacker', 'ensemble'],
        default='ensemble',
        help='Model type to train (default: ensemble)'
    )
    
    parser.add_argument(
        '--features', '-f',
        nargs='+',
        choices=['basic', 'momentum', 'volatility', 'volume', 'patterns', 'advanced'],
        default=None,
        help='Feature groups to use (default: all)'
    )
    
    # Data options
    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date for data (YYYY-MM-DD, default: 2020-01-01)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='End date for data (YYYY-MM-DD, default: 2024-12-31)'
    )
    
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance the dataset (undersample majority class)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='model/multi_stock_predictor.pkl',
        help='Output path for trained model'
    )
    
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Skip fetching data (use existing cached data)'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline for multiple stocks"""
    args = parse_arguments()
    
    # Get ticker list
    if args.preset:
        tickers = STOCK_LISTS[args.preset]
        preset_name = args.preset
    else:
        tickers = [t.upper() for t in args.tickers]
        preset_name = 'custom'
    
    print("="*70)
    print("MULTI-STOCK PREDICTION MODEL TRAINING")
    print("="*70)
    print(f"\nPreset: {preset_name}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"Model Type: {args.model}")
    print(f"Date Range: {args.start} to {args.end}")
    print(f"Feature Groups: {args.features if args.features else 'All'}")
    print(f"Balance Dataset: {args.balance}")
    print(f"Output Path: {args.output}")
    print("="*70)
    
    try:
        # Step 1: Fetch data
        if not args.no_fetch:
            print("\n[1/4] Fetching stock data...")
            successful_tickers, failed_tickers = fetch_multiple_stocks(
                tickers, args.start, args.end
            )
            tickers = successful_tickers
        else:
            print("\n[1/4] Skipping data fetch (using cached data)")
        
        if not tickers:
            raise ValueError("No valid tickers to train on!")
        
        # Step 2: Load and combine data
        print("\n[2/4] Loading and combining data...")
        combined_df, successful_tickers = combine_stock_data(tickers, args.features)
        
        # Step 3: Balance dataset (optional)
        if args.balance:
            print("\n[3/4] Balancing dataset...")
            combined_df = balance_dataset(combined_df)
        else:
            print("\n[3/4] Skipping dataset balancing")
        
        # Remove Ticker column before training
        if 'Ticker' in combined_df.columns:
            combined_df = combined_df.drop('Ticker', axis=1)
        
        # Step 4: Train model
        print(f"\n[4/4] Training {args.model} model on combined data...")
        print(f"Total training samples: {len(combined_df)}")
        
        trainer = ModelTrainer(model_type=args.model)
        accuracy = trainer.train(combined_df, test_size=0.2)
        trainer.save(args.output)
        
        # Save metadata about training
        metadata_path = args.output.replace('.pkl', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Multi-Stock Model Metadata\n")
            f.write(f"="*50 + "\n")
            f.write(f"Trained on: {', '.join(successful_tickers)}\n")
            f.write(f"Date range: {args.start} to {args.end}\n")
            f.write(f"Model type: {args.model}\n")
            f.write(f"Total samples: {len(combined_df)}\n")
            f.write(f"Test accuracy: {accuracy:.2%}\n")
            f.write(f"Balanced: {args.balance}\n")
            f.write(f"Feature groups: {args.features if args.features else 'All'}\n")
        
        print("\n" + "="*70)
        print("✅ MULTI-STOCK TRAINING COMPLETE!")
        print("="*70)
        print(f"Model saved to: {args.output}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Trained on {len(successful_tickers)} stocks: {', '.join(successful_tickers)}")
        print(f"Test Accuracy: {accuracy:.2%}")
        print("\nTo make predictions with this model:")
        print(f"  python predict.py AAPL  # (Will use default model)")
        print(f"  # Or modify predict.py to use: {args.output}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())