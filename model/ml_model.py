import os
import pickle
import json
import pandas as pd
import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from logs.logging import logger

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
os.makedirs(MODEL_DIR, exist_ok=True)
TRAINING_INFO_FILE = os.path.join(MODEL_DIR, "training_info.json")
RETRAIN_INTERVAL_DAYS = 30  # retrain monthly

def get_last_training_date():
    if not os.path.exists(TRAINING_INFO_FILE):
        return None
    with open(TRAINING_INFO_FILE) as f:
        info = json.load(f)
    return pd.to_datetime(info.get("last_training_date")) if info.get("last_training_date") else None

def update_last_training_date(date):
    info = {"last_training_date": str(date.date() if hasattr(date, "date") else date)}
    with open(TRAINING_INFO_FILE, 'w') as f:
        json.dump(info, f)

def needs_retraining():
    last_date = get_last_training_date()
    if last_date is None:
        return True
    days_passed = (pd.Timestamp(datetime.datetime.now().date()) - last_date).days
    return days_passed >= RETRAIN_INTERVAL_DAYS

def prepare_features_and_labels(df: pd.DataFrame, exclude_last_n=3, training=True):
    feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'ma_20', 'ma_50']
    if 'volume' in df.columns:
        feature_cols.append('volume')
    
    df = df.copy().dropna(subset=feature_cols + ['close'])
    
    if training:
        # Create target variable - next day's price movement
        df = df.sort_values(['symbol', 'date'])
        df['next_close'] = df.groupby('symbol')['close'].shift(-1)
        
        # Exclude last n days for each symbol to avoid lookahead bias
        df = df.groupby('symbol').apply(
            lambda x: x.iloc[:-exclude_last_n] if len(x) > exclude_last_n else x
        ).reset_index(drop=True)
        
        # Create target: 1 if next day price goes up, 0 if down
        df['target'] = (df['next_close'] > df['close']).astype(int)
        df = df.dropna(subset=['target'])
        
        X = df[feature_cols]
        y = df['target']
        return X, y, df
    else:
        X = df[feature_cols].copy()
        return X

def train_xgboost_with_validation(X, y):
    """Train XGBoost model with train/test split for validation"""
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Reduced complexity to prevent overfitting
    model = XGBClassifier(
        n_estimators=50,        # Reduced from 150
        max_depth=3,            # Reduced from 5
        learning_rate=0.05,     # Reduced from 0.1
        subsample=0.8,          # Added subsampling
        colsample_bytree=0.8,   # Added feature subsampling
        reg_alpha=1,            # Added L1 regularization
        reg_lambda=1,           # Added L2 regularization
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42
    )
    
    logger.info("Training XGBoost model with regularization...")
    model.fit(X_train, y_train)
    
    # Calculate accuracies
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logger.info(f"Training accuracy: {train_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Overfitting gap: {(train_acc - test_acc):.4f}")
    
    # Detailed classification report
    logger.info("Classification Report on Test Set:")
    logger.info(f"\n{classification_report(y_test, test_pred)}")
    
    return model, train_acc, test_acc

def save_model(model):
    path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Saved XGBoost model to {path}")

def load_model():
    path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    if not os.path.exists(path):
        logger.warning("XGBoost model file not found.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(model, X):
    """Make predictions using the trained model"""
    if model is None:
        return None
    preds = model.predict(X)
    proba = model.predict_proba(X)
    return preds, proba

def translate_prediction(label, confidence_threshold=0.6):
    """Map prediction to trading signal with confidence threshold"""
    # For now, simple mapping - can be enhanced with confidence thresholds
    if label == 1:
        return "BUY"
    elif label == 0:
        return "SELL"
    else:
        return "HOLD"

def get_model_accuracy_on_recent_data(model, df, days_back=30):
    """Test model accuracy on recent N days of data"""
    if model is None:
        return None, None
    
    # Get recent data
    df = df.sort_values(['symbol', 'date'])
    recent_date = df['date'].max() - pd.Timedelta(days=days_back)
    recent_df = df[df['date'] >= recent_date].copy()
    
    if len(recent_df) < 10:  # Need minimum data points
        logger.warning("Insufficient recent data for accuracy testing")
        return None, None
    
    try:
        X_recent, y_recent, _ = prepare_features_and_labels(recent_df, exclude_last_n=1, training=True)
        if len(X_recent) == 0:
            return None, None
            
        predictions = model.predict(X_recent)
        accuracy = accuracy_score(y_recent, predictions)
        
        logger.info(f"Model accuracy on recent {days_back} days: {accuracy:.4f}")
        return accuracy, len(X_recent)
    except Exception as e:
        logger.error(f"Error calculating recent accuracy: {e}")
        return None, None