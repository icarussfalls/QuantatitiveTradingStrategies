import pandas as pd
import numpy as np

def get_latest_signal(stock_a, stock_b, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stop_loss, stock_a_name, stock_b_name):
    # Ensure both dataframes are sorted by date in ascending order
    stock_a = stock_a.sort_values(by='Date', ascending=True)
    stock_b = stock_b.sort_values(by='Date', ascending=True)

    # Align the dates of both stocks using merge
    merged_data = pd.merge(stock_a, stock_b, on='Date', suffixes=('_a', '_b'))

    # Calculate the spread between Stock A and Stock B
    spread = merged_data['LTP_a'] - merged_data['LTP_b']

    # Calculate rolling mean and standard deviation of the spread
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    
    # Calculate z-score of the spread
    z_score = (spread - spread_mean) / spread_std

    # Get the latest z-score
    latest_z_score = z_score.iloc[-1]

    # Generate signal based on the latest z-score
    if latest_z_score < z_entry_thresh_a:
        return f'Enter position: Buy {stock_a_name}. Z-score: {latest_z_score:.2f}'
    elif latest_z_score > z_entry_thresh_b:
        return f'Enter position: Buy {stock_b_name}. Z-score: {latest_z_score:.2f}'
    elif abs(latest_z_score) <= z_exit_thresh:
        return f'Exit position if any. Z-score: {latest_z_score:.2f}'
    else:
        return f'Hold current position or stay out. Z-score: {latest_z_score:.2f}'

# Read CSV file containing parameters
params = pd.read_csv('Day4/results_summary.csv')
print("Parameters DataFrame:\n", params)

pairs_to_trade = [('NYADI', 'NHPC'), ('NYADI', 'DHPL')]

for stock_a_name, stock_b_name in pairs_to_trade:
    # Filter the row corresponding to the stock pair
    row = params[(params['Stock A'] == stock_a_name) & (params['Stock B'] == stock_b_name)]
    
    if not row.empty:
        # Retrieve all parameters for this pair
        rolling_window = int(row['Best Rolling Window'].values[0])
        z_entry_thresh_a = row['Z Entry Threshold A'].values[0]
        z_entry_thresh_b = row['Z Entry Threshold B'].values[0]
        z_exit_thresh = row['Z Exit Threshold'].values[0]
        stop_loss = row['Stop Loss Threshold'].values[0]
        
        print(f"\nParameters for {stock_a_name} and {stock_b_name}:")
        print(f"Rolling Window: {rolling_window}")
        print(f"Z Entry Threshold A: {z_entry_thresh_a:.2f}")
        print(f"Z Entry Threshold B: {z_entry_thresh_b:.2f}")
        print(f"Z Exit Threshold: {z_exit_thresh:.2f}")
        print(f"Stop Loss Threshold: {stop_loss:.2%}")
        
        # Load the stock data (in live trading, this would be the latest data)
        stock_a_data = pd.read_csv(f'datas/{stock_a_name}.csv')
        stock_b_data = pd.read_csv(f'datas/{stock_b_name}.csv')
        
        # Convert 'Date' column to datetime and 'LTP' to numeric
        for df in [stock_a_data, stock_b_data]:
            df['Date'] = pd.to_datetime(df['Date'])
            df['LTP'] = pd.to_numeric(df['LTP'], errors='coerce')
            df.dropna(subset=['LTP'], inplace=True)
        
        # Get the latest signal
        signal = get_latest_signal(stock_a_data, stock_b_data, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stop_loss, stock_a_name, stock_b_name)
        print(signal)
    else:
        print(f"No parameters found for pair {stock_a_name} and {stock_b_name}.")
