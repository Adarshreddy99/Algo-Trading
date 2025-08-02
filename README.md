# Algorithmic Trading System with ML Predictions

A comprehensive algorithmic trading system that combines technical analysis with machine learning predictions for NSE stocks. The system automatically scrapes stock data, generates trading signals, and logs performance metrics to Google Sheets.

## ✨ Features

### 📊 Data Management
- **Automated NSE Data Scraping**: Fetches historical data for RELIANCE, HDFCBANK, and TCS from NSE India
- **Smart Data Updates**: 5-year data updated monthly, 6-month data updated daily
- **Technical Indicators**: Automatically calculates RSI, MACD, 20-DMA, and 50-DMA for all data points
- **Data Persistence**: Stores data in CSV files with technical indicators pre-computed

### 🧠 Machine Learning Model
- **XGBoost Classifier**: Predicts next-day price movements (BUY/SELL signals)
- **Monthly Retraining**: Model retrains automatically every 30 days with latest 5-year data
- **Overfitting Prevention**: Includes regularization, subsampling, and proper train/test splits
- **Performance Tracking**: Displays training accuracy, test accuracy, and recent performance metrics
- **Model Persistence**: Saves trained models as pickle files for daily predictions

### 📈 Trading Strategy
- **Technical Analysis Based**: Uses RSI and moving average crossovers
- **BUY Signal**: RSI < 30 AND 20-DMA > 50-DMA (oversold with upward trend)
- **SELL Signal**: RSI > 70 OR 20-DMA < 50-DMA (overbought or downward trend)
- **HOLD Signal**: When neither BUY nor SELL conditions are met

### 📋 Google Sheets Integration
Three dedicated tabs with automatic updates:

**Trade Log Sheet**:
- Date, Stock Symbol, Strategy Signal, ML Signal, Price, RSI, 20-DMA, 50-DMA

**P&L Summary Sheet**:
- Date, Stock Symbol, P&L % (previous close vs current close comparison)

**Win Ratio Sheet**:
- Stock Symbol, Win Ratio (percentage of days with positive price movements)

### 📊 Backtesting & Analysis
- **6-Month Backtesting**: Tests strategy performance with detailed metrics
- **Signal Analysis**: Shows frequency of trading conditions and signal generation
- **Performance Metrics**: Win rates, total returns, average returns per trade
- **Strategy Verification**: Logs exact conditions that triggered each signal

### 🔧 System Architecture
- **Modular Design**: Separate modules for data loading, ML model, trading strategy, and Google Sheets
- **Comprehensive Logging**: Detailed logs with rotation and console output
- **Error Handling**: Robust error handling for API failures and data issues
- **Environment Configuration**: Uses environment variables for sensitive credentials

## 🔄 Workflow

### Daily Execution Flow
1. **Data Update**: Refreshes 6-month data with latest prices until yesterday
2. **Model Loading**: Loads existing ML model or triggers monthly retraining if needed
3. **Signal Generation**: Generates both technical strategy and ML predictions for today
4. **Performance Analysis**: Runs backtesting and analyzes signal conditions
5. **Prediction Logging**: Updates Google Sheets with today's predictions and analysis
6. **Performance Display**: Shows model accuracy on recent data and trading signals

### Monthly Retraining Flow
1. **5-Year Data Refresh**: Updates complete 5-year dataset
2. **Feature Engineering**: Recalculates all technical indicators
3. **Model Training**: Trains new XGBoost model with regularization
4. **Validation Testing**: Tests on recent data and displays performance metrics
5. **Model Persistence**: Saves new model and updates training timestamp

## 🎯 Target Prediction Logic

The ML model learns to predict next-day price movements:
- **Input Features**: RSI, MACD, MACD Signal, MACD Histogram, 20-DMA, 50-DMA, Volume
- **Target Label**: 1 (BUY) if next day's close > current close, 0 (SELL) otherwise
- **Prediction**: Uses yesterday's technical indicators to predict today's optimal action

## 📊 Performance Metrics

### Model Metrics
- Training vs Test accuracy comparison
- Overfitting gap analysis
- Recent performance on 30-day rolling window
- Classification report with precision, recall, and F1-scores

### Trading Metrics
- Daily win ratio based on price movements
- Backtest results with total returns and win rates
- Signal frequency analysis showing condition occurrence rates
- Strategy logic verification for each prediction

## 🗂️ File Structure

```
├── data/
│   ├── data_loader.py          # NSE data scraping and technical indicators
│   └── data_files/             # CSV storage for historical data
├── model/
│   ├── ml_model.py             # XGBoost training and prediction logic
│   └── models/                 # Saved model files and training info
├── strategy/
│   └── trading_strategy.py     # Technical analysis signal generation
├── sheets/
│   └── sheets_api.py           # Google Sheets integration
├── logs/
│   ├── logging.py              # Logging configuration
│   └── algo_trading.log        # System logs
├── backtest.py                 # Strategy backtesting and analysis
└── main.py                     # Main execution pipeline
```

## 📈 Output Examples

### Daily Predictions
```
RELIANCE: Strategy=BUY, ML=SELL, Price=2,450.75
  RSI=28.45, 20-DMA=2,465.30, 50-DMA=2,420.15, Win Ratio=0.5240
  BUY Logic: RSI(28.45) < 30? True, 20-DMA(2,465.30) > 50-DMA(2,420.15)? True
```

### Backtest Results
```
RELIANCE Backtest Results:
  Total Trades: 45
  Win Rate: 62.22%
  Total Return: 8.45%
  Avg Return/Trade: 0.19%
```

## 🎯 Key Benefits

- **Dual Signal System**: Combines rule-based technical analysis with ML predictions
- **Automated Execution**: Runs daily with minimal manual intervention
- **Performance Tracking**: Comprehensive metrics and backtesting capabilities
- **Data Transparency**: All indicator values and logic displayed in Google Sheets
- **Scalable Architecture**: Easy to add new stocks or modify trading rules
