import pandas as pd
import glob
import os

# Load the file
all_results_df = pd.read_csv('/Users/icarus/Desktop/QuantatitiveTradingStrategies/Pairs Trading/results/cointegrated_pairs_results.csv')

# Select only the required columns
required_columns = [
    'Stock A', 'Stock B', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss', 'Hedge Ratio', 'Half Life'
]

summary_table = all_results_df[required_columns]  # Retain only the specified columns

# Filter for pairs with Sharpe Ratio > 1.5
summary_table = summary_table[summary_table['Final Sharpe Ratio'] > 2]

# Filter for pairs where Average Profit > Average Loss
summary_table = summary_table[summary_table['Average Profit'] > summary_table['Average Loss']]
summary_table = summary_table[summary_table['Win Rate'] > 0.6]


#Round the numeric columns to 2 decimal places
numeric_columns = ['Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss', 'Hedge Ratio', 'Half Life'
]

# Using .loc to avoid SettingWithCopyWarning
summary_table.loc[:, numeric_columns] = summary_table[numeric_columns].round(2)

# Step 9: Filter out rows with inf or 0 values in the numeric columns
summary_table = summary_table[~summary_table[numeric_columns].isin([float('inf'), 0]).any(axis=1)]

# Step 10: Display the summary table with formatted floats
pd.set_option('display.float_format', '{:.2f}'.format)

# Step 11: Print the final summary table
print(summary_table)

filename = "results_summary_filtered.csv"
path = os.path.join(os.getcwd(), 'Pairs Trading/signals/' + filename)
summary_table.to_csv(path, index=False)
