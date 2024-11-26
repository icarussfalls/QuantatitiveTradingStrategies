import pandas as pd
import numpy as np
import logging
from typing import List, Optional

def process_stock_data(
    stock_symbols: List[str], 
    data_directory: str = '/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/', 
    fill_method: Optional[str] = 'ffill'
) -> pd.DataFrame:
    """
    Process stock data ensuring precise date alignment and order.
    
    Args:
        stock_symbols (List[str]): List of stock symbols to process
        data_directory (str): Path to directory containing stock CSV files
        fill_method (Optional[str]): Method to fill missing values
    
    Returns:
        pd.DataFrame: Aligned stock closing prices
    """
    logging.basicConfig(level=logging.INFO)
    
    if len(stock_symbols) < 2:
        logging.error("At least two stock symbols required")
        return pd.DataFrame()

    # Store processed dataframes
    stock_dataframes = {}
    
    for symbol in stock_symbols:
        try:
            # Read stock data
            df = pd.read_csv(
                f'{data_directory}{symbol}.csv', 
                parse_dates=['date'], 
                index_col='date'
            )
            
            # Clean and prepare data
            df.index = pd.to_datetime(df.index).date
            df = df.loc[~df.index.duplicated(keep='first')]
            df = df[df['close'] != 0].dropna(subset=['close'])
            
            stock_dataframes[symbol] = df[['close']]
        
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if len(stock_dataframes) < 2:
        logging.error("Insufficient valid stock data")
        return pd.DataFrame()

    # Find the latest start date and earliest end date
    start_date = max(df.index.min() for df in stock_dataframes.values())
    end_date = min(df.index.max() for df in stock_dataframes.values())

    # Combine data with precise filtering
    combined_df = pd.DataFrame(index=pd.date_range(start_date, end_date))
    for symbol, df in stock_dataframes.items():
        combined_df[symbol] = df.loc[start_date:end_date, 'close']

    # Apply fill method and remove NaN rows
    if fill_method:
        combined_df = combined_df.fillna(method=fill_method)
    combined_df.dropna(inplace=True)
    
    logging.info(f"Processed data for {len(stock_symbols)} stocks")
    return combined_df

def get_latest_signal(merged_data, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name, hedge_ratio):
    """
    Generate trading signal based on Z-score calculations
    """
    # Calculate spread and Z-score
    spread = merged_data[stock_a_name] - hedge_ratio * merged_data[stock_b_name]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    
    z_score = (spread - spread_mean) / spread_std

    # Check for sufficient data
    if len(z_score) < 2:
        return 'Hold', 'None', z_score.iloc[-1] if len(z_score) > 0 else np.nan

    latest_z_score = z_score.iloc[-21]
    previous_z_score = z_score.iloc[-22]
    
    # Signal generation based on Z-score conditions
    if latest_z_score <= z_entry_thresh_a and previous_z_score > z_entry_thresh_a:
        return 'Buy', stock_a_name, latest_z_score
    elif latest_z_score >= z_entry_thresh_b and previous_z_score < z_entry_thresh_b:
        return 'Buy', stock_b_name, latest_z_score
    elif previous_z_score < z_exit_thresh <= latest_z_score:
        return 'Exit', stock_a_name, latest_z_score
    elif previous_z_score > -z_exit_thresh >= latest_z_score:
        return 'Exit', stock_b_name, latest_z_score
    else:
        return 'Hold', 'None', latest_z_score

def run_pairs_trading_strategy(summary_table_path='Pairs Trading/signals/results_summary_filtered.csv'):
    """
    Main function to run pairs trading strategy
    """
    # Load summary table and initialize lists and dictionaries for signal tracking
    summary_table = pd.read_csv(summary_table_path)
    all_signals = []
    buy_count, total_sharpe, total_win_rate = {}, {}, {}
    open_positions = pd.DataFrame(columns=['Stock A', 'Stock B', 'Entry Signal', 'Stock to Buy', 'Entry Z-Score'])

    # Iterate through each stock pair
    for index, row in summary_table.iterrows():
        stock_a_name = row['Stock A']
        stock_b_name = row['Stock B']
        
        # Preprocess stock data
        processed_data = process_stock_data([stock_a_name, stock_b_name])
        
        # Skip if data processing failed
        if processed_data.empty:
            logging.warning(f"Skipping pair {stock_a_name}-{stock_b_name} due to data processing issues")
            continue
        
        # Parameters from summary table
        rolling_window = int(row['Half Life'])
        z_entry_thresh_a, z_entry_thresh_b = row['Z Entry Threshold A'], row['Z Entry Threshold B']
        z_exit_thresh, hedge_ratio = row['Z Exit Threshold'], row['Hedge Ratio']
        
        # Get the latest signal
        signal, stock_to_buy, latest_z_score = get_latest_signal(
            processed_data, 
            rolling_window, 
            z_entry_thresh_a, 
            z_entry_thresh_b, 
            z_exit_thresh, 
            stock_a_name, 
            stock_b_name, 
            hedge_ratio
        )
        
        if signal == 'Buy':
            # Track buy signals and select highest win rate if multiple signals exist
            if stock_to_buy not in buy_count or row['Win Rate'] > total_win_rate.get(stock_to_buy, 0):
                buy_count[stock_to_buy] = buy_count.get(stock_to_buy, 0) + 1
                total_sharpe[stock_to_buy] = row['Final Sharpe Ratio']
                total_win_rate[stock_to_buy] = row['Win Rate']
                open_positions = open_positions.append({
                    'Stock A': stock_a_name,
                    'Stock B': stock_b_name,
                    'Entry Signal': signal,
                    'Stock to Buy': stock_to_buy,
                    'Entry Z-Score': latest_z_score
                }, ignore_index=True)
        elif signal == 'Exit':
            # Check for exit signals against open positions
            exits = open_positions[(open_positions['Stock A'] == stock_a_name) & (open_positions['Stock B'] == stock_b_name)]
            if not exits.empty:
                all_signals.append({
                    'Stock A': stock_a_name,
                    'Stock B': stock_b_name,
                    'Signal': signal,
                    'Stock to Sell': stock_to_buy,
                    'Exit Z-Score': latest_z_score
                })
                open_positions = open_positions[~((open_positions['Stock A'] == stock_a_name) & (open_positions['Stock B'] == stock_b_name))]

    # Ranking based on win rate and Sharpe ratio
    ranking_df = pd.DataFrame({
        'Stock': list(buy_count.keys()),
        'Buy Count': list(buy_count.values()),
        'Average Sharpe Ratio': [total_sharpe[stock] for stock in buy_count.keys()],
        'Win Rate': [total_win_rate[stock] for stock in buy_count.keys()],
        'Half Life': [
            summary_table.loc[summary_table['Stock A'] == stock, 'Half Life'].values[0] if stock in summary_table['Stock A'].values else 
            summary_table.loc[summary_table['Stock B'] == stock, 'Half Life'].values[0] if stock in summary_table['Stock B'].values else 
            None
            for stock in buy_count.keys()
        ]
    })

    ranking_df.sort_values(by=['Half Life', 'Average Sharpe Ratio', 'Win Rate'], ascending=[True, False, False], inplace=True)
    print("\nRanked Stocks to Buy by Half Life:")
    print(ranking_df)

    # Save ranked stocks and signals
    ranking_df.to_csv("Pairs Trading/signals/ranked_stocks_to_buy_by_half_life.csv", index=False)
    open_positions.to_csv("Pairs Trading/signals/open_positions.csv", index=False)
    print("\nSaved open positions and ranking data.")

    return ranking_df, open_positions

# Main execution
if __name__ == "__main__":
    run_pairs_trading_strategy()