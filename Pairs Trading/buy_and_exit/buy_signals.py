import pandas as pd
import numpy as np
import glob
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
    
    # Avoid division by zero in Z-score calculation by adding a small constant to spread_std
    spread_std = spread_std.replace(0, np.nan)  # Convert 0 std to NaN for safe calculation
    z_score = (spread - spread_mean) / spread_std

    # Drop any NaN Z-scores
    z_score = z_score.dropna()
    
    # Check for sufficient Z-score data to generate signal
    if len(z_score) < 2:
        return 'Hold', 'None', z_score.iloc[-1] if len(z_score) > 0 else np.nan  # Handle case with insufficient data

    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2]
    print("Latest Z-score:", latest_z_score, "Previous Z-score:", previous_z_score)

    # Generate signal based on Z-score conditions
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


# Get the summary table saved from the save_best_signals py
summary_table = pd.read_csv('Pairs Trading/signals/results_summary_filtered.csv')

# Prepare a list to hold all signals
all_signals = []

# Initialize dictionaries for buy count, total Sharpe ratio, and total win rate
buy_count = {}
total_sharpe = {}
total_win_rate = {}

# Iterate through each stock pair in the summary table
for index, row in summary_table.iterrows():
    stock_a_name = row['Stock A']
    stock_b_name = row['Stock B']
    
    # Retrieve all parameters for this pair
    rolling_window = int(row['Half Life'])
    z_entry_thresh_a = row['Z Entry Threshold A']
    z_entry_thresh_b = row['Z Entry Threshold B']
    z_exit_thresh = row['Z Exit Threshold']
    hedge_ratio = row['Hedge Ratio']  # Now correctly using the hedge ratio from the results

    # Load the stock data
    stock_a_data = pd.read_csv(f'datas/{stock_a_name}.csv')
    stock_b_data = pd.read_csv(f'datas/{stock_b_name}.csv')
    
    # Convert 'Date' column to datetime and 'LTP' to numeric
    for df in [stock_a_data, stock_b_data]:
        df['Date'] = pd.to_datetime(df['date'])
        df['LTP'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(subset=['LTP'], inplace=True)
    
    # Get the latest signal
    signal, stock_to_buy, latest_z_score = get_latest_signal(stock_a_data, stock_b_data, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name, hedge_ratio)
    
    # Append the results to the signals list
    all_signals.append({
        'Stock A': stock_a_name,
        'Stock B': stock_b_name,
        'Signal': signal,
        'Stock to Buy': stock_to_buy,
        'Latest Z-Score': latest_z_score,
        'Rolling Window': rolling_window,
        'Z Entry Threshold A': z_entry_thresh_a,
        'Z Entry Threshold B': z_entry_thresh_b,
        'Z Exit Threshold': z_exit_thresh
    })

    # Count buy signals for the stocks
    if signal == 'Buy':
        if stock_to_buy not in buy_count:
            buy_count[stock_to_buy] = 0
            total_sharpe[stock_to_buy] = 0
            total_win_rate[stock_to_buy] = 0
        buy_count[stock_to_buy] += 1

        # Accumulate total Sharpe ratio and win rate
        total_sharpe[stock_to_buy] += row['Final Sharpe Ratio']
        total_win_rate[stock_to_buy] += row['Win Rate']

# Step 16: Create a DataFrame for average Sharpe ratio, win rate, and half-life
ranking_df = pd.DataFrame({
    'Stock': buy_count.keys(),
    'Buy Count': buy_count.values(),
    'Average Sharpe Ratio': [total_sharpe[stock] / buy_count[stock] for stock in buy_count.keys()],
    'Win Rate': [total_win_rate[stock] / buy_count[stock] for stock in buy_count.keys()],
    'Half Life': [
        summary_table.loc[summary_table['Stock A'] == stock, 'Half Life'].values[0] if stock in summary_table['Stock A'].values else 
        summary_table.loc[summary_table['Stock B'] == stock, 'Half Life'].values[0] if stock in summary_table['Stock B'].values else 
        None  # Use None instead of 0 to indicate no half-life found
        for stock in buy_count.keys()
    ]
})

# Step 17: Sort the ranking DataFrame by Half Life, then Average Sharpe Ratio, and then Win Rate
ranking_df.sort_values(by=['Half Life', 'Average Sharpe Ratio', 'Win Rate'], ascending=[True, False, False], inplace=True)

# Step 18: Display the ranking DataFrame
print("\nRanked Stocks to Buy by Half Life:")
print(ranking_df)

# Save ranking DataFrame to CSV
filename = "ranked_stocks_to_buy_by_half_life.csv"
path = os.path.join(os.getcwd(), 'Pairs Trading/signals/' + filename)
ranking_df.to_csv(path, index=False)
