import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from logs.logging import logger
from strategy.trading_strategy import generate_signals

def run_backtest(df: pd.DataFrame, months_back=6):
    """
    Backtest the trading strategy over the specified period
    """
    logger.info(f"Running backtest for {months_back} months")
    
    # Filter data for backtest period
    end_date = df['date'].max()
    start_date = end_date - pd.Timedelta(days=months_back * 30)
    backtest_df = df[df['date'] >= start_date].copy()
    
    results = {}
    total_trades = 0
    total_profit = 0
    
    for stock in backtest_df['symbol'].unique():
        stock_data = backtest_df[backtest_df['symbol'] == stock].sort_values('date').copy()
        
        if len(stock_data) < 10:  # Need minimum data
            continue
            
        # Generate signals
        stock_data = generate_signals(stock_data)
        
        # Simulate trading
        position = None  # None, 'BUY', 'SELL'
        entry_price = 0
        trades = []
        
        for i in range(len(stock_data) - 1):  # Exclude last day
            current_signal = stock_data.iloc[i]['signal']
            current_price = stock_data.iloc[i]['close']
            next_price = stock_data.iloc[i + 1]['open']  # Next day opening
            
            if position is None and current_signal in ['BUY', 'SELL']:
                # Enter position
                position = current_signal
                entry_price = next_price
                
            elif position is not None and current_signal != position:
                # Exit position
                if position == 'BUY':
                    profit_pct = ((next_price - entry_price) / entry_price) * 100
                else:  # position == 'SELL'
                    profit_pct = ((entry_price - next_price) / entry_price) * 100
                
                trades.append({
                    'entry_date': stock_data.iloc[i-1]['date'] if i > 0 else stock_data.iloc[i]['date'],
                    'exit_date': stock_data.iloc[i]['date'],
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': next_price,
                    'profit_pct': profit_pct
                })
                
                position = current_signal if current_signal != 'HOLD' else None
                entry_price = next_price if position else 0
        
        # Calculate statistics
        if trades:
            profits = [t['profit_pct'] for t in trades]
            win_trades = [p for p in profits if p > 0]
            
            results[stock] = {
                'total_trades': len(trades),
                'winning_trades': len(win_trades),
                'win_rate': len(win_trades) / len(trades) * 100,
                'total_return': sum(profits),
                'avg_return_per_trade': np.mean(profits),
                'best_trade': max(profits) if profits else 0,
                'worst_trade': min(profits) if profits else 0,
                'trades': trades
            }
            
            total_trades += len(trades)
            total_profit += sum(profits)
            
            logger.info(f"{stock} Backtest Results:")
            logger.info(f"  Total Trades: {len(trades)}")
            logger.info(f"  Win Rate: {len(win_trades) / len(trades) * 100:.2f}%")
            logger.info(f"  Total Return: {sum(profits):.2f}%")
            logger.info(f"  Avg Return/Trade: {np.mean(profits):.2f}%")
    
    # Overall results
    if total_trades > 0:
        logger.info(f"\nOverall Backtest Results ({months_back} months):")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Total Return: {total_profit:.2f}%")
        logger.info(f"Avg Return/Trade: {total_profit/total_trades:.2f}%")
    
    return results

def analyze_signal_conditions(df: pd.DataFrame):
    """
    Analyze how often each signal condition is met
    """
    logger.info("Analyzing Trading Signal Conditions:")
    
    for stock in df['symbol'].unique():
        stock_data = df[df['symbol'] == stock].copy()
        stock_data = generate_signals(stock_data)
        
        # Count signal conditions
        buy_rsi_condition = (stock_data['rsi'] < 30).sum()
        buy_ma_condition = (stock_data['ma_20'] > stock_data['ma_50']).sum()
        buy_both_condition = ((stock_data['rsi'] < 30) & (stock_data['ma_20'] > stock_data['ma_50'])).sum()
        
        sell_rsi_condition = (stock_data['rsi'] > 70).sum()
        sell_ma_condition = (stock_data['ma_20'] < stock_data['ma_50']).sum()
        sell_either_condition = ((stock_data['rsi'] > 70) | (stock_data['ma_20'] < stock_data['ma_50'])).sum()
        
        buy_signals = (stock_data['signal'] == 'BUY').sum()
        sell_signals = (stock_data['signal'] == 'SELL').sum()
        hold_signals = (stock_data['signal'] == 'HOLD').sum()
        
        total_days = len(stock_data)
        
        logger.info(f"\n{stock} Signal Analysis ({total_days} days):")
        logger.info(f"  RSI < 30: {buy_rsi_condition} days ({buy_rsi_condition/total_days*100:.1f}%)")
        logger.info(f"  20-DMA > 50-DMA: {buy_ma_condition} days ({buy_ma_condition/total_days*100:.1f}%)")
        logger.info(f"  BUY conditions (both): {buy_both_condition} days ({buy_both_condition/total_days*100:.1f}%)")
        logger.info(f"  RSI > 70: {sell_rsi_condition} days ({sell_rsi_condition/total_days*100:.1f}%)")
        logger.info(f"  20-DMA < 50-DMA: {sell_ma_condition} days ({sell_ma_condition/total_days*100:.1f}%)")
        logger.info(f"  SELL conditions (either): {sell_either_condition} days ({sell_either_condition/total_days*100:.1f}%)")
        logger.info(f"  Final Signals - BUY: {buy_signals}, SELL: {sell_signals}, HOLD: {hold_signals}")
        
        # Latest values
        latest = stock_data.iloc[-1]
        logger.info(f"  Latest Values - RSI: {latest['rsi']:.2f}, 20-DMA: {latest['ma_20']:.2f}, 50-DMA: {latest['ma_50']:.2f}")