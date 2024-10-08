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

    # Get the last two z-scores
    if len(z_score) < 2:
        return 'Hold', 'None', z_score.iloc[-1]  # Not enough data to decide

    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2]

    # Generate signal based on the latest z-score
    if latest_z_score < z_entry_thresh_a:  # Buy Stock A when z-score is below its entry threshold
        return 'Buy', stock_a_name, latest_z_score
    elif latest_z_score > z_entry_thresh_b:  # Buy Stock B when z-score is above its entry threshold
        return 'Buy', stock_b_name, latest_z_score
    elif previous_z_score < z_exit_thresh <= latest_z_score:  # Exit signal for Stock A
        return 'Exit', stock_a_name, latest_z_score
    elif previous_z_score > -z_exit_thresh >= latest_z_score:  # Exit signal for Stock B
        return 'Exit', stock_b_name, latest_z_score
    else:
        return 'Hold', 'None', latest_z_score  # Hold current position


# Step 1: Load all CSV files from the results directory
file_path_pattern = 'Pairs Trading/results/*.csv'  # Correct wildcard to match all CSV files
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

# Step 7: Filter for pairs where Average Profit > Average Loss
summary_table = summary_table[summary_table['Average Profit'] > summary_table['Average Loss']]

# Step 9: Round the numeric columns to 2 decimal places
numeric_columns = [
    'Best Rolling Window', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]

# Using .loc to avoid SettingWithCopyWarning
summary_table.loc[:, numeric_columns] = summary_table[numeric_columns].round(2)

# Step 9: Filter out rows with inf or 0 values in the numeric columns
summary_table = summary_table[~summary_table[numeric_columns].isin([float('inf'), 0]).any(axis=1)]

# Step 10: Display the summary table with formatted floats
pd.set_option('display.float_format', '{:.2f}'.format)

# Step 11: Print the final summary table
print(summary_table)

# Step 12: Optionally save to a new CSV
filename = "results_summary_filtered.csv"
path = os.path.join(os.getcwd(), 'Pairs Trading/signals/' + filename)
summary_table.to_csv(path, index=False)

# Prepare a list to hold all signals
all_signals = []

# Initialize dictionaries for buy count, profitability, win rate, and Sharpe ratio
buy_count = {}
average_sharpe = {}
win_rate = {}

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
        df['LTP'] = pd.to_numeric(df['Close'], errors='coerce')
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

        # Track average Sharpe ratio and win rate
        if stock_to_buy not in average_sharpe:
            average_sharpe[stock_to_buy] = row['Final Sharpe Ratio']
            win_rate[stock_to_buy] = row['Win Rate']
        else:
            # Update average Sharpe ratio
            average_sharpe[stock_to_buy] = (average_sharpe[stock_to_buy] + row['Final Sharpe Ratio']) / 2  # Average Sharpe ratio
            win_rate[stock_to_buy] = (win_rate[stock_to_buy] + row['Win Rate']) / 2  # Average win rate

# Step 12: Convert signals list to DataFrame
signals_df = pd.DataFrame(all_signals)

# Step 13: Save the signals DataFrame to a CSV file
filename = 'generated_signals.csv'
path = os.path.join(os.getcwd(), 'Pairs Trading/signals/' + filename)
signals_df.to_csv(path, index=False)

# Step 14: Print the count of buy signals for each stock
print("Buy Count for Each Stock:")
for stock, count in buy_count.items():
    print(f"{stock}: {count}")

# Step 15: Create a DataFrame for average Sharpe ratio and win rate
ranking_df = pd.DataFrame({
    'Stock': buy_count.keys(),
    'Buy Count': buy_count.values(),
    'Average Sharpe Ratio': [average_sharpe[stock] for stock in buy_count.keys()],
    'Win Rate': [win_rate[stock] for stock in buy_count.keys()]
})

# Step 16: Sort the ranking DataFrame by Average Sharpe Ratio and Win Rate
ranking_df.sort_values(by=['Average Sharpe Ratio', 'Win Rate'], ascending=False, inplace=True)

# Step 17: Display the ranking DataFrame
print("\nRanked Stocks to Buy:")
print(ranking_df)

filename = "ranked_stocks_to_buy.csv"
# Save DataFrame to CSV
path = os.path.join(os.getcwd(), 'Pairs Trading/signals/' + filename)
ranking_df.to_csv(path, index=False)
