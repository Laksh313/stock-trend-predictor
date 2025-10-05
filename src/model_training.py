from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import joblib
import os

class ModelTrainer:
    """Unified trainer for multiple model types"""
    
    def __init__(self, model_type='ensemble'):
        """
        Initialize trainer with model type
        
        Args:
            model_type: 'random_forest', 'lightgbm', 'lstm', 'stacker', 'ensemble'
        """
        self.model_type = model_type
        self.model = None
        self.lstm_model = None
        self.feature_cols = None
        
    def prepare_data(self, df, feature_cols=None):
        """Prepare features and target"""
        if feature_cols is None:
            feature_cols = [
                "Open", "High", "Low", "Close", "Volume",
                "SMA_10", "SMA_50", "EMA_10", "Volatility",
                "RSI", "MACD", "MACD_Signal"
            ]
        
        self.feature_cols = feature_cols
        
        for col in feature_cols + ["Target"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        X = df[feature_cols].copy()
        y = df["Target"].copy()
        
        X = X.dropna()
        y = y.loc[X.index]
        
        return X, y
    
    def create_lstm_sequences(self, X, y, seq_length=10):
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        X_arr = X.values
        y_arr = y.values
        
        for i in range(len(X_arr) - seq_length):
            X_seq.append(X_arr[i:i+seq_length])
            y_seq.append(y_arr[i+seq_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM"""
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=8,
            num_leaves=31,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            force_col_wise=True
        )
        model.fit(X_train, y_train)
        return model
    
    def train_lstm(self, X_train, y_train, X_test, y_test, seq_length=10):
        """Train LSTM model"""
        X_train_seq, y_train_seq = self.create_lstm_sequences(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            seq_length
        )
        X_test_seq, y_test_seq = self.create_lstm_sequences(
            X_test.reset_index(drop=True),
            y_test.reset_index(drop=True),
            seq_length
        )
        
        model = self.build_lstm_model((seq_length, X_train.shape[1]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=100,
            batch_size=64,
            callbacks=[early_stop],
            verbose=0
        )
        
        return model, (X_test_seq, y_test_seq)
    
    def train_stacker(self, X_train, y_train):
        """Train Stacking Ensemble"""
        from sklearn.ensemble import GradientBoostingClassifier
        from xgboost import XGBClassifier
        
        try:
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=300, 
                    max_depth=20,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )),
                ('lgb', LGBMClassifier(
                    n_estimators=300, 
                    learning_rate=0.01,
                    max_depth=8,
                    class_weight='balanced',
                    random_state=42, 
                    verbose=-1
                )),
                ('xgb', XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.01,
                    max_depth=8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ))
            ]
        except ImportError:
            # Fallback if XGBoost not installed
            estimators = [
                ('rf', RandomForestClassifier(
                    n_estimators=300, 
                    max_depth=20,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )),
                ('lgb', LGBMClassifier(
                    n_estimators=300, 
                    learning_rate=0.01,
                    max_depth=8,
                    class_weight='balanced',
                    random_state=42, 
                    verbose=-1
                )),
                ('gb', GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.01,
                    max_depth=8,
                    random_state=42
                ))
            ]
        
        model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000
            ),
            cv=5,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train multiple models and ensemble predictions"""
        print("Training Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)
        
        print("Training LightGBM...")
        lgb_model = self.train_lightgbm(X_train, y_train)
        
        print("Training Stacker...")
        stack_model = self.train_stacker(X_train, y_train)
        
        self.model = {
            'random_forest': rf_model,
            'lightgbm': lgb_model,
            'stacker': stack_model
        }
        
        return self.model
    
    def train(self, df, test_size=0.2):
        """Main training method"""
        X, y = self.prepare_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        print(f"\nTraining {self.model_type} model...")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        if self.model_type == 'random_forest':
            self.model = self.train_random_forest(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
        elif self.model_type == 'lightgbm':
            self.model = self.train_lightgbm(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
        elif self.model_type == 'lstm':
            self.lstm_model, (X_test_seq, y_test_seq) = self.train_lstm(
                X_train, y_train, X_test, y_test
            )
            y_pred_proba = self.lstm_model.predict(X_test_seq)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            y_test = y_test_seq
            
        elif self.model_type == 'stacker':
            self.model = self.train_stacker(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
        elif self.model_type == 'ensemble':
            self.model = self.train_ensemble(X_train, y_train, X_test, y_test)
            # Ensemble prediction (majority voting)
            predictions = []
            for name, model in self.model.items():
                predictions.append(model.predict(X_test))
            y_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        print(f"\n✅ Model trained. Test Accuracy: {acc:.2%}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
        
        return acc
    
    def save(self, model_path):
        """Save model(s) to disk"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        if self.model_type == 'lstm':
            # Save LSTM model separately
            lstm_path = model_path.replace('.pkl', '_lstm.h5')
            self.lstm_model.save(lstm_path)
            # Save metadata
            metadata = {
                'model_type': self.model_type,
                'feature_cols': self.feature_cols,
                'lstm_path': lstm_path
            }
            joblib.dump(metadata, model_path)
            print(f"LSTM model saved to {lstm_path}")
        else:
            # Save sklearn models
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_cols': self.feature_cols
            }
            joblib.dump(model_data, model_path)
        
        print(f"Model metadata saved to {model_path}")


def train_model(df, model_path, model_type='ensemble'):
    """
    Convenience function for backward compatibility
    
    Args:
        df: DataFrame with features and target
        model_path: Path to save model
        model_type: Type of model ('random_forest', 'lightgbm', 'lstm', 'stacker', 'ensemble')
    """
    trainer = ModelTrainer(model_type=model_type)
    trainer.train(df)
    trainer.save(model_path)