import pandas as pd
import numpy as np

def check_exit(half_life, days_passed, z_score_threshold, up):
    # up checks if stock is located on stock A or stock B

    if half_life < days_passed:
        print('exit the trade as half life has passed')
        return
        # exit the trade as the pairs have converged
    if up:
        if z_score_threshold[-2] > z_score_threshold and z_score_threshold[-1] < z_score_threshold:
            print('exit signal generated for stock located in up')
            return
    if not up:
        if z_score_threshold[-2] < z_score_threshold and z_score_threshold[-1] > z_score_threshold:
            print('exit signal generated for stock located for down')
            return
    else:
        print('no exit signal has been generated')
        return []
    

# lets get the values now
current_positions = ['NYADI']
# open the csv
summary_table = pd.read_csv('Pairs Trading/signals/results_summary_filtered.csv')

# read the table
for stock_display_name in current_positions:
    # Load the corresponding stock data
    stock_data = pd.read_csv(f'datas/{stock_display_name}.csv')
    print(stock_data)

# get the summmary table values for which stock is located
# should we record when buy I think yes


