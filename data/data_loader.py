import os
import time
import random
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

import pandas as pd
import requests
from logs.logging import logger

DATA_DIR = Path(__file__).resolve().parent.parent / "data_files"
DATA_DIR.mkdir(exist_ok=True)

DATE_FMT = "%d-%m-%Y"
CHUNK_SIZE_MONTHS = 2
MAX_RETRIES = 4

NSE_BASE = "https://www.nseindia.com"
HIST_ENDPOINT = NSE_BASE + "/api/historical/cm/equity"

STOCKS = ["RELIANCE", "HDFCBANK", "TCS"]

FILE_5Y = DATA_DIR / "combined_5y.csv"
FILE_6M = DATA_DIR / "combined_6m.csv"


def calculate_rsi(prices: pd.Series, window=14) -> pd.Series:
    """Calculate RSI for given price series"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window, min_periods=1).mean()

def calculate_macd(prices: pd.Series, fast=12, slow=26, signal=9):
    """Calculate MACD indicators"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, 20 DMA, 50 DMA, and MACD to the dataframe"""
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])
    
    # Group by symbol and calculate indicators
    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df[mask].copy()
        
        # Calculate technical indicators
        symbol_data['rsi'] = calculate_rsi(symbol_data['close'])
        symbol_data['ma_20'] = calculate_sma(symbol_data['close'], 20)
        symbol_data['ma_50'] = calculate_sma(symbol_data['close'], 50)
        
        # Calculate MACD
        macd_line, signal_line, macd_hist = calculate_macd(symbol_data['close'])
        symbol_data['macd'] = macd_line
        symbol_data['macd_signal'] = signal_line
        symbol_data['macd_hist'] = macd_hist
        
        # Update the original dataframe
        df.loc[mask, ['rsi', 'ma_20', 'ma_50', 'macd', 'macd_signal', 'macd_hist']] = symbol_data[['rsi', 'ma_20', 'ma_50', 'macd', 'macd_signal', 'macd_hist']].values
    
    return df


def _make_headers(referer=None):
    user_agent = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/605.1.15 Version/16.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125.0.0.0 Safari/537.36"
    ])
    return {
        "User-Agent": user_agent,
        "Accept": "application/json, text/plain, */*",
        "Referer": referer or f"{NSE_BASE}/get-quotes/equity",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
        "X-Requested-With": "XMLHttpRequest",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
    }


def _prime_session(session, symbol):
    try:
        session.get(f"{NSE_BASE}/get-quotes/equity?symbol={symbol}", headers=_make_headers())
    except Exception:
        pass
    time.sleep(random.uniform(0.3, 1.0))
    try:
        session.get(NSE_BASE, headers=_make_headers())
    except Exception:
        pass
    time.sleep(random.uniform(0.5, 1.5))


def fetch_data_chunk(session, symbol, start_date, end_date):
    params = {
        "symbol": symbol,
        "series": '["EQ"]',
        "from": start_date,
        "to": end_date
    }
    for attempt in range(MAX_RETRIES):
        response = session.get(HIST_ENDPOINT, params=params, headers=_make_headers())
        if response.status_code == 200:
            try:
                data = response.json()
                if data.get("data"):
                    return data
            except Exception:
                time.sleep(2 ** attempt + random.random())
                continue
        elif response.status_code in (401, 403):
            _prime_session(session, symbol)
        time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"Failed to fetch data from {start_date} to {end_date} for {symbol}")


def scrape_data_for_symbol(session, symbol, months):
    data_rows = []
    seen = set()
    start = datetime.today() - relativedelta(months=months)
    end = datetime.today()
    cursor = start
    while cursor < end:
        nxt = min(end, cursor + relativedelta(months=CHUNK_SIZE_MONTHS))
        data = fetch_data_chunk(session, symbol, cursor.strftime(DATE_FMT), nxt.strftime(DATE_FMT))
        for row in data.get("data", []):
            key = (row.get("date"), row.get("CH_OPENING_PRICE"), row.get("CH_CLOSING_PRICE"))
            if key not in seen:
                seen.add(key)
                row["symbol"] = symbol
                data_rows.append(row)
        cursor = nxt + relativedelta(days=1)
    return data_rows


def load_or_update_raw_data(months, file_path, force_update=False):
    # Check existing file for data up to yesterday
    yesterday = datetime.today().date() - pd.Timedelta(days=1)
    
    if not force_update and file_path.exists():
        df = pd.read_csv(file_path, parse_dates=["date"])
        if df["date"].max().date() >= yesterday:
            logger.info(f"{file_path.name} is up to date – skipping scrape.")
            # Add technical indicators if not present
            if 'rsi' not in df.columns:
                df = add_technical_indicators(df)
                df.to_csv(file_path, index=False)
                logger.info(f"Added technical indicators to {file_path.name}")
            return df

    session = requests.Session()
    all_data = []
    for symbol in STOCKS:
        _prime_session(session, symbol)
        rows = scrape_data_for_symbol(session, symbol, months)
        df_sym = pd.DataFrame(rows)
        df_sym.rename(columns=lambda c: c.lower(), inplace=True)
        df_sym.rename(columns={
            "ch_opening_price": "open",
            "ch_closing_price": "close",
            "ch_trade_high_price": "high",
            "ch_trade_low_price": "low",
            "ch_total_traded_quantity": "volume"
        }, inplace=True)
        
        # Ensure date column exists and is parsed
        if "date" not in df_sym.columns:
            date_col = next((c for c in df_sym.columns if "date" in c.lower()), None)
            if date_col:
                df_sym["date"] = pd.to_datetime(df_sym[date_col], errors='coerce')
        else:
            df_sym["date"] = pd.to_datetime(df_sym["date"], errors='coerce')

        df_sym = df_sym[["date", "symbol", "open", "high", "low", "close"]]
        all_data.append(df_sym)

    df_all = pd.concat(all_data).drop_duplicates(subset=["symbol", "date"])
    df_all.sort_values(["symbol", "date"], inplace=True)
    
    # Add technical indicators
    df_all = add_technical_indicators(df_all)
    
    df_all.to_csv(file_path, index=False)
    logger.info(f"Saved {file_path.name} with {len(df_all)} rows and technical indicators")
    return df_all


def update_5y_data(force_update=False):
    """Update 5 year data - only when forced (monthly retraining)"""
    return load_or_update_raw_data(60, FILE_5Y, force_update)


def update_6m_data():
    """Update 6 month data daily"""
    # Always check for new daily data for 6 month file
    yesterday = datetime.today().date() - pd.Timedelta(days=1)
    
    if FILE_6M.exists():
        df_6m = pd.read_csv(FILE_6M, parse_dates=["date"])
        if df_6m["date"].max().date() >= yesterday:
            logger.info(f"{FILE_6M.name} is up to date")
            return df_6m
    
    # Get updated data for last 6 months
    df_new = load_or_update_raw_data(6, DATA_DIR / "temp_6m.csv", force_update=True)
    
    # Save as 6m file
    df_new.to_csv(FILE_6M, index=False)
    logger.info(f"Updated {FILE_6M.name} for daily prediction")
    
    # Clean up temp file
    temp_file = DATA_DIR / "temp_6m.csv"
    if temp_file.exists():
        temp_file.unlink()
    
    return df_new


def add_new_day_to_data(new_rows: pd.DataFrame):
    """Append new daily rows for today to both files."""
    for fp in (FILE_5Y, FILE_6M):
        if fp.exists():
            df = pd.read_csv(fp, parse_dates=['date'])
            df = pd.concat([df, new_rows], ignore_index=True)
            df.drop_duplicates(subset=['date', 'symbol'], keep='last', inplace=True)
            # Recalculate technical indicators for updated data
            df = add_technical_indicators(df)
        else:
            df = add_technical_indicators(new_rows.copy())
        df.to_csv(fp, index=False)