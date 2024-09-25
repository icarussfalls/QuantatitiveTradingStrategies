import pandas as pd

# Step 1: Load the CSV file
file_path = 'results.csv'  # Ensure this path is correct
df = pd.read_csv(file_path)

# Step 2: Select only the required columns
required_columns = [
    'Stock A', 'Stock B', 'Best Rolling Window', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Cumulative Returns Start', 'Cumulative Returns End',  # Added 'Cumulative Returns Start'
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]

summary_table = df[required_columns]  # Retain only the specified columns

# Step 3: Round the numeric columns to 2 decimal places
numeric_columns = [
    'Best Rolling Window', 'Z Entry Threshold A', 'Z Entry Threshold B',
    'Z Exit Threshold', 'Stop Loss Threshold', 'Final Cumulative Returns',
    'Final Sharpe Ratio', 'Final Sortino Ratio', 'Final Max Drawdown', 
    'Cumulative Returns Start', 'Cumulative Returns End',  # Include rounding for all relevant numeric columns
    'Total Trades', 'Win Rate', 'Profit Factor', 'Average Profit', 'Average Loss'
]

# Using .loc to avoid SettingWithCopyWarning
summary_table.loc[:, numeric_columns] = summary_table[numeric_columns].round(2)

# Step 4: Display the summary table with formatted floats
pd.set_option('display.float_format', '{:.2f}'.format)

# Step 5: Print the final summary table
print(summary_table)

# Step 6: Optionally save to a new CSV
summary_table.to_csv('results_summary.csv', index=False)
