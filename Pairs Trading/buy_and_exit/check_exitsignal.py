import pandas as pd
import numpy as np
import os

def check_exit_signal(stock_a, stock_b, rolling_window, z_exit_thresh, hedge_ratio):
    stock_a = stock_a.sort_values(by='date', ascending=True)
    stock_b = stock_b.sort_values(by='date', ascending=True)
    merged_data = pd.merge(stock_a, stock_b, on='date', suffixes=('_a', '_b'))

    spread = merged_data['LTP_a'] - hedge_ratio * merged_data['LTP_b']
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std().replace(0, np.nan)
    z_score = (spread - spread_mean) / spread_std
    z_score = z_score.dropna()

    if len(z_score) < 2:
        return 'Hold', np.nan

    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2]

    if previous_z_score < z_exit_thresh <= latest_z_score or previous_z_score > -z_exit_thresh >= latest_z_score:
        return 'Exit', latest_z_score
    return 'Hold', latest_z_score

# Load positions and check for exit signals
positions_db_path = 'Pairs Trading/signals/positions_db.csv'
positions = pd.read_csv(positions_db_path)

for _, position in positions.iterrows():
    stock_a = pd.read_csv(f'datas/{position["Stock"]}.csv')
    stock_b = pd.read_csv(f'datas/{position["Stock"]}.csv')
    
    # Get exit signal
    signal, latest_z_score = check_exit_signal(stock_a, stock_b, rolling_window=20, z_exit_thresh=1, hedge_ratio=0.5)
    if signal == 'Exit':
        exit_data = {'Stock': position['Stock'], 'Z-Score': latest_z_score}
        
        # Append to exits database
        exits_db_path = 'Pairs Trading/signals/exits_db.csv'
        pd.DataFrame([exit_data]).to_csv(exits_db_path, mode='a', header=not os.path.exists(exits_db_path), index=False)
        print(f"Exit signal generated and saved for {position['Stock']}.")
