import pandas as pd
import os

def get_latest_signal(stock_a, stock_b, rolling_window, hedge_ratio):
    # Ensure both dataframes are sorted by date in ascending order
    stock_a = stock_a.sort_values(by='Date', ascending=True)
    stock_b = stock_b.sort_values(by='Date', ascending=True)

    # Align the dates of both stocks using merge
    merged_data = pd.merge(stock_a, stock_b, on='Date', suffixes=('_a', '_b'))
    # Calculate the spread between Stock A and Stock B using the hedge ratio
    spread = merged_data['LTP_a'] - hedge_ratio * merged_data['LTP_b']

    # Calculate rolling mean and standard deviation of the spread
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    
    # Calculate z-score of the spread
    z_score = (spread - spread_mean) / spread_std

    # Get the latest z-score and the previous z-score
    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2] if len(z_score) > 1 else None

    return latest_z_score, previous_z_score

# Define your current positions in a list format
current_positions = ['PRSF']  # Add more stocks as needed

# Prepare a list to hold exit signals
exit_signals = []

# Load the summary table to find relevant pairs
summary_table = pd.read_csv('Pairs Trading/signals/results_summary_filtered.csv')

# Iterate through each stock in the current positions
for stock_display_name in current_positions:
    # Load the corresponding stock data
    stock_data = pd.read_csv(f'datas/{stock_display_name}.csv')

    # Filter the summary table for pairs that include the current stock
    relevant_pairs = summary_table[(summary_table['Stock A'] == stock_display_name) | (summary_table['Stock B'] == stock_display_name)]

    for index, row in relevant_pairs.iterrows():
        # Determine the other stock in the pair
        other_stock_name = row['Stock B'] if row['Stock A'] == stock_display_name else row['Stock A']

        # Retrieve all parameters for this pair
        rolling_window = int(row['Half Life'])
        z_exit_thresh = row['Z Exit Threshold']
        hedge_ratio = row['Hedge Ratio']

        # Load the other stock's data
        other_stock_data = pd.read_csv(f'datas/{other_stock_name}.csv')

        # Convert 'Date' column to datetime and 'LTP' to numeric for both stocks
        for df in [stock_data, other_stock_data]:
            df['Date'] = pd.to_datetime(df['date'])
            df['LTP'] = pd.to_numeric(df['close'], errors='coerce')
            df.dropna(subset=['LTP'], inplace=True)

        # Get the latest z-score for this pair
        latest_z_score_a, previous_z_score_a = get_latest_signal(stock_data, other_stock_data, rolling_window, hedge_ratio)

        # Check for exit signals based on z-score crossings
        exit_signal = None

        # Check if Stock A's z-score crossed from below to above the exit threshold
        if previous_z_score_a is not None and previous_z_score_a < z_exit_thresh and latest_z_score_a >= z_exit_thresh:
            exit_signal = {
                'Stock A': stock_display_name,
                'Stock B': other_stock_name,
                'Signal': 'Exit',
                'Latest Z-Score': latest_z_score_a,
                'Rolling Window': rolling_window,
                'Z Exit Threshold': z_exit_thresh
            }

        # Check if Stock B's z-score crossed from above to below the exit threshold
        # Calculate Stock B's latest and previous z-scores
        latest_z_score_b, previous_z_score_b = get_latest_signal(other_stock_data, stock_data, rolling_window, hedge_ratio)
        if previous_z_score_b is not None and previous_z_score_b > z_exit_thresh and latest_z_score_b <= z_exit_thresh:
            exit_signal = {
                'Stock A': other_stock_name,
                'Stock B': stock_display_name,
                'Signal': 'Exit',
                'Latest Z-Score': latest_z_score_b,
                'Rolling Window': rolling_window,
                'Z Exit Threshold': z_exit_thresh
            }

        # Append exit signal information if applicable
        if exit_signal:
            exit_signals.append(exit_signal)

# Convert exit signals list to DataFrame
exit_signals_df = pd.DataFrame(exit_signals)

# Save the exit signals DataFrame to a CSV file

filename = "exit_signals.csv"
# Save DataFrame to CSV
path = os.path.join(os.getcwd(), 'Pairs Trading/signals/' + filename)
exit_signals_df.to_csv(path, index=False)

# Print the exit signals
print("Exit Signals:")
print(exit_signals_df)
