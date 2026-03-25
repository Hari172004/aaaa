import os
import logging
import pandas as pd
import numpy as np
from ml.signal_classifier import SignalClassifier # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window).mean()

def retrain_from_csv(symbol: str = "XAUUSD"):
    csv_path = f"data/{symbol}_D1_history.csv"
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return

    logger.info(f"Loading {symbol} history from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Feature Engineering
    df['rsi'] = calculate_rsi(df['close'])
    df['ema_21'] = df['close'].ewm(span=21).mean()
    df['ema_diff'] = (df['close'] - df['ema_21']) / df['close'] * 100
    df['atr'] = calculate_atr(df)
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Rename features to match SignalClassifier expected names
    # Features in SignalClassifier: ["rsi", "ema_distance", "atr", "volume_ratio", "session_id", "news_score", "spread", "mtf_confluence"]
    df = df.rename(columns={
        'ema_diff': 'ema_distance',
        'vol_ratio': 'volume_ratio'
    })
    
    # Fill placeholders
    df['session_id'] = 2 
    df['news_score'] = 0.0
    df['spread'] = 1.0
    df['mtf_confluence'] = 3
    
    # Labeling: 1 if price moves +1.5 ATR in 3 days, -1 if -1.5 ATR, else 0
    df['result'] = 0
    horizon = 3
    for i in range(len(df) - horizon):
        future_change = df['close'].iloc[i+horizon] - df['close'].iloc[i]
        threshold = df['atr'].iloc[i] * 1.5
        if future_change > threshold:
            df.loc[df.index[i], 'result'] = 1
        elif future_change < -threshold:
            df.loc[df.index[i], 'result'] = -1

    df.dropna(inplace=True)
    
    # Train model
    classifier = SignalClassifier(symbol=symbol)
    success, accuracy = classifier.train_model(df)
    
    if success:
        logger.info(f"Retraining {symbol} SUCCESS. Accuracy: {accuracy*100:.2f}%")
    else:
        logger.error(f"Retraining {symbol} FAILED. Accuracy: {accuracy*100:.2f}%")
        
    return success, accuracy

if __name__ == "__main__":
    # Train Gold
    retrain_from_csv("XAUUSD")
    
    # Train BTC
    retrain_from_csv("BTCUSD")
