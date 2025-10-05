import argparse
from data_fetch import fetch_stock_data
from data_preprocessing import load_and_clean
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train stock price prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train ensemble model with all features
  python main.py AAPL --model ensemble
  
  # Train LightGBM with specific features
  python main.py AAPL --model lightgbm --features basic momentum volatility
  
  # Train LSTM with custom date range
  python main.py AAPL --model lstm --start 2020-01-01 --end 2024-12-31
        """
    )
    
    parser.add_argument(
        'ticker',
        type=str,
        help='Stock ticker symbol (e.g., AAPL, MSFT, TSLA)'
    )
    
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
        '--output', '-o',
        type=str,
        default='model/stock_predictor.pkl',
        help='Output path for trained model (default: model/stock_predictor.pkl)'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_arguments()
    
    ticker = args.ticker.upper()
    model_type = args.model
    feature_groups = args.features
    start_date = args.start
    end_date = args.end
    model_path = args.output
    
    print("="*60)
    print("STOCK PRICE PREDICTION MODEL TRAINING")
    print("="*60)
    print(f"\nTicker: {ticker}")
    print(f"Model Type: {model_type}")
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Feature Groups: {feature_groups if feature_groups else 'All'}")
    print(f"Output Path: {model_path}")
    print("="*60)
    
    try:
        # Step 1: Fetch stock data
        print("\n[1/4] Fetching stock data...")
        data_path = f"data/{ticker}_data.csv"
        fetch_stock_data(ticker, start_date, end_date)
        
        # Step 2: Load and clean data
        print("\n[2/4] Loading and cleaning data...")
        df = load_and_clean(data_path)
        print(f"Loaded {len(df)} rows of data")
        
        # Step 3: Engineer features
        print("\n[3/4] Engineering features...")
        engineer = FeatureEngineer()
        df_features = engineer.engineer_features(df, feature_groups)
        
        feature_cols = [col for col in df_features.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
        print(f"Created {len(feature_cols)} features")
        print(f"Final dataset shape: {df_features.shape}")
        
        # Step 4: Train model
        print(f"\n[4/4] Training {model_type} model...")
        trainer = ModelTrainer(model_type=model_type)
        
        # Set custom feature columns if needed
        if feature_groups:
            trainer.feature_cols = feature_cols
        
        accuracy = trainer.train(df_features)
        trainer.save(model_path)
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print("="*60)
        print(f"Model saved to: {model_path}")
        print(f"Test Accuracy: {accuracy:.2%}")
        print("\nTo make predictions, run:")
        print(f"  python predict.py {ticker}")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())