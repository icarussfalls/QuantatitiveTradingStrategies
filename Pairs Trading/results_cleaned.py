import pandas as pd
import glob
import os

def get_latest_signal(stock_a, stock_b, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name):
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
    if latest_z_score < z_entry_thresh_a:  # Buy Stock A when z-score is below its entry threshold
        return 'Buy', stock_a_name, latest_z_score
    elif latest_z_score > z_entry_thresh_b:  # Buy Stock B when z-score is above its entry threshold
        return 'Buy', stock_b_name, latest_z_score
    elif latest_z_score > z_exit_thresh + 0.02 or latest_z_score < -z_exit_thresh - 0.02:  # Exit signal for both conditions
        return 'Exit', 'None', latest_z_score
    else:
        return 'Hold', 'None', latest_z_score  # Hold current position


# Step 1: Load all CSV files from the results directory
file_path_pattern = 'results/*.csv'  # Correct wildcard to match all CSV files
all_files = glob.glob(file_path_pattern)

# Step 2: Create an empty list to store DataFrames
df_list = []

# Step 3: Loop through each file and append its DataFrame to the list
for file in all_files:
    df = pd.read_csv(file)
    df_list.append(df)

# Step 4: Concatenate all DataFrames into a single DataFrame
all_results_df = pd.concat(df_list, ignore_index=True)

# Step 5: Select only the required columns
required_columns = [
    'Stock A', 'Stock B', 'Best Rolling Window', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]

summary_table = all_results_df[required_columns]  # Retain only the specified columns

# Step 6: Filter for pairs with Sharpe Ratio > 2
summary_table = summary_table[summary_table['Final Sharpe Ratio'] > 2]

# Step 7: Round the numeric columns to 2 decimal places
numeric_columns = [
    'Best Rolling Window', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]

# Using .loc to avoid SettingWithCopyWarning
summary_table.loc[:, numeric_columns] = summary_table[numeric_columns].round(2)

# Step 8: Filter out rows with inf or 0 values in the numeric columns
summary_table = summary_table[~summary_table[numeric_columns].isin([float('inf'), 0]).any(axis=1)]

# Step 9: Display the summary table with formatted floats
pd.set_option('display.float_format', '{:.2f}'.format)

# Step 10: Print the final summary table
print(summary_table)

# Step 11: Optionally save to a new CSV
summary_table.to_csv('results_summary_filtered.csv', index=False)

# Prepare a list to hold all signals
all_signals = []

# Count of buy signals for each stock
buy_count = {}

# Iterate through each stock pair in the summary table
for index, row in summary_table.iterrows():
    stock_a_name = row['Stock A']
    stock_b_name = row['Stock B']
    
    # Retrieve all parameters for this pair
    rolling_window = int(row['Best Rolling Window'])
    z_entry_thresh_a = row['Z Entry Threshold A']
    z_entry_thresh_b = row['Z Entry Threshold B']
    z_exit_thresh = row['Z Exit Threshold']
    
    # Load the stock data
    stock_a_data = pd.read_csv(f'datas/{stock_a_name}.csv')
    stock_b_data = pd.read_csv(f'datas/{stock_b_name}.csv')
    
    # Convert 'Date' column to datetime and 'LTP' to numeric
    for df in [stock_a_data, stock_b_data]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['LTP'] = pd.to_numeric(df['LTP'], errors='coerce')
        df.dropna(subset=['LTP'], inplace=True)
    
    # Get the latest signal
    signal, stock_to_buy, latest_z_score = get_latest_signal(stock_a_data, stock_b_data, rolling_window, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name)
    
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
        buy_count[stock_to_buy] += 1

# Step 12: Convert signals list to DataFrame
signals_df = pd.DataFrame(all_signals)

# Step 13: Save the signals DataFrame to a CSV file
signals_df.to_csv('generated_signals.csv', index=False)

# Step 14: Print the count of buy signals for each stock
print("Buy Count for Each Stock:")
for stock, count in buy_count.items():
    print(f"{stock}: {count}")
