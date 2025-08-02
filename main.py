import os
import pandas as pd
from datetime import datetime, timedelta
from data.data_loader import update_5y_data, update_6m_data, STOCKS
from model.ml_model import (
    prepare_features_and_labels,
    train_xgboost_with_validation, 
    save_model, 
    load_model, 
    needs_retraining, 
    update_last_training_date,
    translate_prediction,
    predict,
    get_model_accuracy_on_recent_data
)
from sheets.sheets_api import (
    connect_to_sheet, 
    log_trade_signal, 
    update_pnl_summary, 
    update_win_ratio
)
from logs.logging import logger
from strategy.trading_strategy import generate_signals
from backtest import run_backtest, analyze_signal_conditions

def calculate_pnl_percentage(prev_close, current_open):
    """Calculate P&L percentage between previous close and current open"""
    if prev_close and current_open and prev_close != 0:
        return round(((current_open - prev_close) / prev_close) * 100, 2)
    return 0.0

def calculate_win_ratio(df_stock):
    """Calculate win ratio based on daily price movements"""
    df_stock = df_stock.sort_values('date')
    daily_returns = df_stock['close'].pct_change()
    wins = (daily_returns > 0).sum()
    total_days = len(daily_returns.dropna())
    return round(wins / total_days if total_days > 0 else 0, 4)

def main():
    logger.info("=== Algorithmic Trading Pipeline Started ===")
    
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    
    # Step 1: Data Management
    # Check if we need monthly retraining (update 5Y data only then)
    if needs_retraining():
        logger.info("Monthly retraining required - updating 5 year data")
        df_5y = update_5y_data(force_update=True)
    else:
        logger.info("Using existing 5 year data for model")
        df_5y = update_5y_data(force_update=False)
    
    # Always update 6 month data daily
    logger.info("Updating 6 month data for daily predictions")
    df_6m = update_6m_data()
    
    # Step 2: Connect to Google Sheets
    sheet = None
    try:
        sheet = connect_to_sheet()
        logger.info("Successfully connected to Google Sheets")
    except Exception as e:
        logger.error(f"Failed to connect to Google Sheets: {e}")
        logger.info("Continuing without Google Sheets integration")

    # Step 3: Model Training/Loading
    model = None
    if needs_retraining():
        logger.info("Training new model with 5 years of data")
        try:
            X, y, train_df = prepare_features_and_labels(df_5y, training=True)
            if len(X) > 0:
                model, train_acc, test_acc = train_xgboost_with_validation(X, y)
                save_model(model)
                update_last_training_date(datetime.now())
                logger.info(f"Model trained successfully - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            else:
                logger.error("No training data available")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    else:
        logger.info("Loading existing model")
        model = load_model()
        if model is None:
            logger.info("No existing model found, training new model")
            try:
                X, y, train_df = prepare_features_and_labels(df_5y, training=True)
                if len(X) > 0:
                    model, train_acc, test_acc = train_xgboost_with_validation(X, y)
                    save_model(model)
                    update_last_training_date(datetime.now())
                    logger.info(f"New model trained - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
            except Exception as e:
                logger.error(f"Model training failed: {e}")

    # Step 4: Test model accuracy on recent data
    if model is not None:
        recent_acc, test_points = get_model_accuracy_on_recent_data(model, df_6m, days_back=30)
        if recent_acc is not None:
            logger.info(f"Model performance on recent 30 days: {recent_acc:.4f} ({test_points} data points)")

    # Step 5: Analyze strategy and run backtest
    logger.info("=== Strategy Analysis ===")
    analyze_signal_conditions(df_6m)
    
    logger.info("=== Running 6-Month Backtest ===")
    backtest_results = run_backtest(df_6m, months_back=6)
    
    # Step 6: Generate predictions for today
    logger.info(f"Generating predictions for {today}")
    
    # Convert date column for filtering
    df_6m['date'] = pd.to_datetime(df_6m['date']).dt.date
    
    # Prepare data for logging
    trade_rows = []
    pnl_rows = []
    win_ratios = {}
    
    for stock in STOCKS:
        logger.info(f"Processing {stock}")
        
        # Get stock data
        df_stock = df_6m[df_6m['symbol'] == stock].sort_values('date')
        
        # Get yesterday's data for prediction
        yesterday_data = df_stock[df_stock['date'] == yesterday]
        
        if yesterday_data.empty:
            logger.warning(f"No data available for {stock} on {yesterday}")
            continue
        
        latest_row = yesterday_data.iloc[-1]
        
        # Generate strategy signal using trading_strategy logic
        strategy_signal = 'HOLD'
        if len(yesterday_data) > 0:
            # Create a proper DataFrame for strategy calculation
            strategy_df = yesterday_data.copy()
            strategy_result = generate_signals(strategy_df)
            strategy_signal = strategy_result['signal'].iloc[0]
        
        # Generate ML prediction
        ml_signal = 'HOLD'
        if model is not None:
            try:
                feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'ma_20', 'ma_50']
                if 'volume' in df_stock.columns:
                    feature_cols.append('volume')
                
                # Get features from yesterday's data
                features = latest_row[feature_cols].values.reshape(1, -1)
                
                # Check for NaN values in features
                if not pd.isna(features).any():
                    predictions, probabilities = predict(model, features)
                    ml_signal = translate_prediction(predictions[0])
                else:
                    logger.warning(f"NaN values in features for {stock}, using HOLD signal")
            except Exception as e:
                logger.error(f"ML prediction failed for {stock}: {e}")
        
        # Prepare trade log entry with indicator values
        trade_rows.append([
            today.strftime("%Y-%m-%d"),
            stock,
            strategy_signal,
            ml_signal,
            latest_row['close'],
            round(latest_row.get('rsi', 0), 2),
            round(latest_row.get('ma_20', 0), 2),
            round(latest_row.get('ma_50', 0), 2)
        ])
        
        # Calculate P&L (previous close vs today's opening)
        # Since we don't have today's opening yet, we'll use yesterday's close as approximation
        # In real implementation, this would be updated with actual opening prices
        pnl_date = yesterday  # P&L is for yesterday, not today
        if len(df_stock) >= 2:
            prev_row = df_stock.iloc[-2]  # Day before yesterday
            pnl_pct = calculate_pnl_percentage(prev_row['close'], latest_row['open'])
        else:
            pnl_pct = 0.0
        
        pnl_rows.append([
            pnl_date.strftime("%Y-%m-%d"),  # Use yesterday's date for P&L
            stock,
            pnl_pct
        ])
        
        # Calculate win ratio for the stock
        win_ratios[stock] = calculate_win_ratio(df_stock)
        
        logger.info(f"{stock}: Strategy={strategy_signal}, ML={ml_signal}, Price={latest_row['close']:.2f}")
        logger.info(f"  RSI={latest_row.get('rsi', 0):.2f}, 20-DMA={latest_row.get('ma_20', 0):.2f}, 50-DMA={latest_row.get('ma_50', 0):.2f}, Win Ratio={win_ratios[stock]:.4f}")
        
        # Verify trading strategy logic
        rsi_val = latest_row.get('rsi', 0)
        ma20_val = latest_row.get('ma_20', 0)
        ma50_val = latest_row.get('ma_50', 0)
        
        if strategy_signal == 'BUY':
            logger.info(f"  BUY Logic: RSI({rsi_val:.2f}) < 30? {rsi_val < 30}, 20-DMA({ma20_val:.2f}) > 50-DMA({ma50_val:.2f})? {ma20_val > ma50_val}")
        elif strategy_signal == 'SELL':
            logger.info(f"  SELL Logic: RSI({rsi_val:.2f}) > 70? {rsi_val > 70}, 20-DMA({ma20_val:.2f}) < 50-DMA({ma50_val:.2f})? {ma20_val < ma50_val}")
        else:
            logger.info(f"  HOLD Logic: No clear BUY/SELL conditions met")

    # Step 8: Update Google Sheets
    if sheet and (trade_rows or pnl_rows or win_ratios):
        try:
            if trade_rows:
                log_trade_signal(sheet, trade_rows)
                logger.info("Trade signals logged to Google Sheets")
            
            if pnl_rows:
                update_pnl_summary(sheet, pnl_rows)
                logger.info("P&L summary updated in Google Sheets")
            
            if win_ratios:
                update_win_ratio(sheet, win_ratios)
                logger.info("Win ratios updated in Google Sheets")
                
        except Exception as e:
            logger.error(f"Failed to update Google Sheets: {e}")
    else:
        if not sheet:
            logger.warning("Google Sheets not connected - data not uploaded")
        else:
            logger.info("No data to upload to Google Sheets")

    # Step 9: Summary
    logger.info("=== Pipeline Summary ===")
    logger.info(f"Date: {today}")
    logger.info(f"Stocks processed: {len(STOCKS)}")
    logger.info(f"Trade signals generated: {len(trade_rows)}")
    logger.info(f"Model status: {'Loaded' if model else 'Not available'}")
    
    if trade_rows:
        logger.info("Today's Signals:")
        for row in trade_rows:
            logger.info(f"  {row[1]}: Strategy={row[2]}, ML={row[3]}, Price={row[4]}, RSI={row[5]}, 20-DMA={row[6]}, 50-DMA={row[7]}")
    
    logger.info("=== Algorithmic Trading Pipeline Completed ===")


if __name__ == "__main__":
    main()