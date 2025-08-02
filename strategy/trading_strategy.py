import pandas as pd

def calculate_rsi(prices: pd.Series, window=14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=window).mean()

def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['ma_20'] = calculate_sma(df['close'], 20)
    df['ma_50'] = calculate_sma(df['close'], 50)
    macd_line, signal_line, macd_hist = calculate_macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist

    df['signal'] = 'HOLD'
    df.loc[(df['rsi'] < 30) & (df['ma_20'] > df['ma_50']), 'signal'] = 'BUY'
    df.loc[(df['rsi'] > 70) | (df['ma_20'] < df['ma_50']), 'signal'] = 'SELL'
    return df