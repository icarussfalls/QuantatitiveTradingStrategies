import pandas as pd
import numpy as np
import os

def get_latest_signal(stock_a, stock_b, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name, hedge_ratio):
    # Sort and merge data
    stock_a = stock_a.sort_values(by='date', ascending=True)
    stock_b = stock_b.sort_values(by='date', ascending=True)
    merged_data = pd.merge(stock_a, stock_b, on='date', suffixes=('_a', '_b'))
    
    # Calculate spread and Z-score
    spread = merged_data['LTP_a'] - hedge_ratio * merged_data['LTP_b']
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    spread_std = spread_std.replace(0, np.nan)  # Avoid division by zero
    
    z_score = (spread - spread_mean) / spread_std
    z_score = z_score.dropna()  # Drop NaN Z-scores

    # Check for sufficient data
    if len(z_score) < 2:
        return 'Hold', 'None', z_score.iloc[-1] if len(z_score) > 0 else np.nan

    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2]
    
    # Signal generation based on Z-score conditions
    if latest_z_score < z_entry_thresh_a and previous_z_score > z_entry_thresh_a:
        return 'Buy', stock_a_name, latest_z_score
    elif latest_z_score > z_entry_thresh_b and previous_z_score < z_entry_thresh_b:
        return 'Buy', stock_b_name, latest_z_score
    elif previous_z_score < z_exit_thresh <= latest_z_score:
        return 'Exit', stock_a_name, latest_z_score
    elif previous_z_score > -z_exit_thresh >= latest_z_score:
        return 'Exit', stock_b_name, latest_z_score
    else:
        return 'Hold', 'None', latest_z_score

# Load summary table and initialize lists and dictionaries for signal tracking
summary_table = pd.read_csv('Pairs Trading/signals/results_summary_filtered.csv')
all_signals = []
buy_count, total_sharpe, total_win_rate = {}, {}, {}
open_positions = pd.DataFrame(columns=['Stock A', 'Stock B', 'Entry Signal', 'Stock to Buy', 'Entry Z-Score'])

# Iterate through each stock pair
for index, row in summary_table.iterrows():
    stock_a_name = row['Stock A']
    stock_b_name = row['Stock B']
    
    rolling_window = int(row['Half Life'])
    z_entry_thresh_a, z_entry_thresh_b = row['Z Entry Threshold A'], row['Z Entry Threshold B']
    z_exit_thresh, hedge_ratio = row['Z Exit Threshold'], row['Hedge Ratio']

    stock_a_data = pd.read_csv(f'datas/{stock_a_name}.csv')
    stock_b_data = pd.read_csv(f'datas/{stock_b_name}.csv')
    
    for df in [stock_a_data, stock_b_data]:
        df['Date'] = pd.to_datetime(df['date'])
        df['LTP'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['LTP'], inplace=True)
    
    # Get the latest signal
    signal, stock_to_buy, latest_z_score = get_latest_signal(stock_a_data, stock_b_data, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name, hedge_ratio)
    
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
    'Stock': buy_count.keys(),
    'Buy Count': buy_count.values(),
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
