"""
Compare performance of single-stock vs multi-stock models
"""

import sys
import joblib
import yfinance as yf
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

def load_model(model_path):
    """Load a trained model"""
    if not os.path.exists(model_path):
        return None
    
    model_data = joblib.load(model_path)
    return model_data

def fetch_and_prepare_data(ticker, days=250):
    """Fetch and prepare data for testing"""
    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    df.dropna(inplace=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    return df_features

def evaluate_model(model_data, df, model_name):
    """Evaluate a model on test data"""
    feature_cols = model_data['feature_cols']
    model = model_data['model']
    model_type = model_data['model_type']
    
    # Prepare features and target
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Make predictions
    if model_type == 'ensemble':
        predictions = []
        for name, m in model.items():
            pred = m.predict(X)
            predictions.append(pred)
        y_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    else:
        y_pred = model.predict(X)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'actual': y
    }

def compare_models(ticker, single_model_path='model/stock_predictor.pkl', 
                   multi_model_path='model/multi_stock_predictor.pkl'):
    """Compare single-stock and multi-stock models"""
    
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON FOR {ticker}")
    print(f"{'='*70}\n")
    
    # Load models
    print("Loading models...")
    single_model = load_model(single_model_path)
    multi_model = load_model(multi_model_path)
    
    if single_model is None and multi_model is None:
        print("❌ No models found! Train a model first.")
        return
    
    # Fetch data
    print(f"Fetching recent data for {ticker}...")
    df = fetch_and_prepare_data(ticker)
    
    # Split into test set (last 20%)
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[-test_size:]
    
    print(f"Test set size: {len(df_test)} samples\n")
    
    results = []
    
    # Evaluate single-stock model
    if single_model:
        print("Evaluating single-stock model...")
        result = evaluate_model(single_model, df_test, "Single-Stock")
        results.append(result)
    else:
        print(f"⚠ Single-stock model not found at {single_model_path}")
    
    # Evaluate multi-stock model
    if multi_model:
        print("Evaluating multi-stock model...")
        result = evaluate_model(multi_model, df_test, "Multi-Stock")
        results.append(result)
    else:
        print(f"⚠ Multi-stock model not found at {multi_model_path}")
    
    # Display comparison
    if results:
        print(f"\n{'='*70}")
        print("COMPARISON RESULTS")
        print(f"{'='*70}\n")
        
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['model']:<20} "
                  f"{result['accuracy']:<12.2%} "
                  f"{result['precision']:<12.2%} "
                  f"{result['recall']:<12.2%} "
                  f"{result['f1']:<12.2%}")
        
        print()
        
        # Determine winner
        if len(results) == 2:
            if results[1]['accuracy'] > results[0]['accuracy']:
                improvement = (results[1]['accuracy'] - results[0]['accuracy']) * 100
                print(f"🏆 Multi-stock model performs better by {improvement:.2f}%")
            elif results[0]['accuracy'] > results[1]['accuracy']:
                improvement = (results[0]['accuracy'] - results[1]['accuracy']) * 100
                print(f"🏆 Single-stock model performs better by {improvement:.2f}%")
            else:
                print("📊 Both models have equal accuracy")
        
        # Show detailed breakdown
        print(f"\n{'='*70}")
        print("PREDICTION BREAKDOWN")
        print(f"{'='*70}\n")
        
        for result in results:
            actual_up = np.sum(result['actual'])
            pred_up = np.sum(result['predictions'])
            correct = np.sum(result['predictions'] == result['actual'])
            
            print(f"{result['model']} Model:")
            print(f"  Correct predictions: {correct}/{len(result['actual'])} ({correct/len(result['actual'])*100:.1f}%)")
            print(f"  Predicted UP: {pred_up} | Actual UP: {actual_up}")
            print(f"  Predicted DOWN: {len(result['actual'])-pred_up} | Actual DOWN: {len(result['actual'])-actual_up}")
            print()
        
        print(f"{'='*70}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_models.py <TICKER> [single_model_path] [multi_model_path]")
        print("\nExamples:")
        print("  python compare_models.py AAPL")
        print("  python compare_models.py TSLA model/custom_single.pkl model/custom_multi.pkl")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    
    single_path = sys.argv[2] if len(sys.argv) > 2 else 'model/stock_predictor.pkl'
    multi_path = sys.argv[3] if len(sys.argv) > 3 else 'model/multi_stock_predictor.pkl'
    
    try:
        compare_models(ticker, single_path, multi_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()