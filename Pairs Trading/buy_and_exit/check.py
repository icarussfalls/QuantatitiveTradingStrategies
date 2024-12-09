import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize

def process_stock_data(
    stock_symbols: List[str], 
    data_directory: str = '/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/', 
    fill_method: Optional[str] = 'ffill'
) -> pd.DataFrame:
    """
    Process stock data ensuring precise date alignment and order.
    
    Args:
        stock_symbols (List[str]): List of stock symbols to process
        data_directory (str): Path to directory containing stock CSV files
        fill_method (Optional[str]): Method to fill missing values
    
    Returns:
        pd.DataFrame: Aligned stock closing prices
    """
    logging.basicConfig(level=logging.INFO)
    
    if len(stock_symbols) < 2:
        logging.error("At least two stock symbols required")
        return pd.DataFrame()

    # Store processed dataframes
    stock_dataframes = {}
    
    for symbol in stock_symbols:
        try:
            # Read stock data
            df = pd.read_csv(
                f'{data_directory}{symbol}.csv', 
                parse_dates=['date'], 
                index_col='date'
            )
            
            # Clean and prepare data
            df.index = pd.to_datetime(df.index).date
            df = df.loc[~df.index.duplicated(keep='first')]
            df = df[df['close'] != 0].dropna(subset=['close'])
            
            stock_dataframes[symbol] = df[['close']]
        
        except Exception as e:
            logging.error(f"Error processing {symbol}: {e}")
    
    if len(stock_dataframes) < 2:
        logging.error("Insufficient valid stock data")
        return pd.DataFrame()

    # Find the latest start date and earliest end date
    start_date = max(df.index.min() for df in stock_dataframes.values())
    end_date = min(df.index.max() for df in stock_dataframes.values())

    # Combine data with precise filtering
    combined_df = pd.DataFrame(index=pd.date_range(start_date, end_date))
    for symbol, df in stock_dataframes.items():
        combined_df[symbol] = df.loc[start_date:end_date, 'close']

    # Apply fill method and remove NaN rows
    if fill_method:
        combined_df = combined_df.fillna(method=fill_method)
    combined_df.dropna(inplace=True)
    
    logging.info(f"Processed data for {len(stock_symbols)} stocks")
    return combined_df

# Function to calculate hedge ratio
def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio

# Calculate OU params
def OU(spread):
    """
    Fit an Ornstein-Uhlenbeck process to the given spread data.
    Returns estimated parameters: theta, mu, csigma.
    """
    # Define the OU process objective function
    def ou_objective(params):
        theta, mu, sigma = params
        dt = 1  # Daily data, so time step is 1 day
        spread_diff = spread.diff().dropna()
        spread_lag = spread.shift(1).dropna()
        
        # OU model: dS = theta * (mu - S) * dt + sigma * dW
        predicted_diff = theta * (mu - spread_lag) * dt
        residual = spread_diff - predicted_diff
        
        # Minimize the squared error (residuals)
        return np.sum(residual**2)
    
    # Initial guess for the parameters [theta, mu, sigma]
    initial_guess = [0.1, spread.mean(), spread.std()]
    
    # Minimize the objective function to estimate the parameters
    result = minimize(ou_objective, initial_guess, bounds=[(0, None), (None, None), (0, None)])
    
    # Extract the fitted parameters
    theta, mu, sigma = result.x
    return theta, mu, sigma

def half_life_calc(theta):
    return np.log(2)/theta


# Function to calculate half-life only once
def get_half_life(stock_a, stock_b, data):
    """
    Retrieve the cached half-life for stock pair or calculate if not available.
    """
    pair = tuple(sorted([stock_a, stock_b]))
    spread = data[stock_a] - calculate_hedge_ratio(data[stock_a], data[stock_b]) * data[stock_b]
    theta, _, _ = OU(spread)
    if not np.isfinite(theta) or theta <= 0:
        logging.warning(f"Skipping pair {stock_a}-{stock_b} due to invalid theta.")
        return None  # Return None for invalid half-life
    half_life = np.log(2) / theta
    return half_life


def get_latest_signal(merged_data, z_entry_thresh_a, z_entry_thresh_b, z_exit_thresh, stock_a_name, stock_b_name):
    """
    Generate trading signal based on Z-score calculations
    """
    # Calculate spread and Z-score
    half_life = get_half_life(stock_a_name, stock_b_name, merged_data)
    if half_life is None or half_life < 5:
        half_life = 5
    
    # Set rolling window based on the half-life
    rolling_window = int(half_life)

    # Calculate the hedge ratio
    hedge_ratio = calculate_hedge_ratio(merged_data[stock_a_name], merged_data[stock_b_name])

    # Calculate spread
    spread = merged_data[stock_a_name] - hedge_ratio * merged_data[stock_b_name]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    
    z_score = (spread - spread_mean) / spread_std

    # Check for sufficient data
    if len(z_score) < 2:
        return 'Hold', 'None', z_score.iloc[-1] if len(z_score) > 0 else np.nan

    latest_z_score = z_score.iloc[-1]
    previous_z_score = z_score.iloc[-2]
    
    # Signal generation based on Z-score conditions
    if latest_z_score <= z_entry_thresh_a and previous_z_score > z_entry_thresh_a:
        return 'Buy', stock_a_name, latest_z_score
    elif latest_z_score >= z_entry_thresh_b and previous_z_score < z_entry_thresh_b:
        return 'Buy', stock_b_name, latest_z_score
    elif previous_z_score < z_exit_thresh <= latest_z_score:
        return 'Exit', stock_a_name, latest_z_score
    elif previous_z_score > -z_exit_thresh >= latest_z_score:
        return 'Exit', stock_b_name, latest_z_score
    else:
        return 'Hold', 'None', latest_z_score

def run_pairs_trading_strategy(summary_table_path='Pairs Trading/signals/results_summary_filtered.csv', output_path='signals_output.csv'):
    """
    Main function to run pairs trading strategy and save signals
    """
    # Load summary table and initialize signal tracking
    summary_table = pd.read_csv(summary_table_path)
    all_signals = []
    buy_count, total_sharpe, total_win_rate = {}, {}, {}
    open_positions = pd.DataFrame(columns=['Stock A', 'Stock B', 'Entry Signal', 'Stock to Buy', 'Entry Z-Score', 'Half-Life'])

    # Iterate through each stock pair
    for index, row in summary_table.iterrows():
        stock_a_name = row['Stock A']
        stock_b_name = row['Stock B']
        
        # Preprocess stock data
        processed_data = process_stock_data([stock_a_name, stock_b_name])
        # Use the last 100 data points
        processed_data = processed_data[-252:]

        if processed_data.empty:
            logging.warning(f"Skipping pair {stock_a_name}-{stock_b_name} due to data processing issues")
            continue
        
        # Calculate half-life and ensure validity
        half_life = get_half_life(stock_a_name, stock_b_name, processed_data)
        if half_life is None or half_life < 5:
            half_life = 5  # Set a minimum half-life threshold for calculations
        
        # Parameters from summary table
        z_entry_thresh_a, z_entry_thresh_b = row['Z Entry Threshold A'], row['Z Entry Threshold B']
        z_exit_thresh = row['Z Exit Threshold']
        
        # Get the latest signal
        signal, stock_to_buy, latest_z_score = get_latest_signal(
            processed_data, 
            z_entry_thresh_a, 
            z_entry_thresh_b, 
            z_exit_thresh, 
            stock_a_name, 
            stock_b_name
        )
        
        # Track signals
        if signal == 'Buy':
            if stock_to_buy not in buy_count or row['Win Rate'] > total_win_rate.get(stock_to_buy, 0):
                buy_count[stock_to_buy] = buy_count.get(stock_to_buy, 0) + 1
                total_sharpe[stock_to_buy] = row['Final Sharpe Ratio']
                total_win_rate[stock_to_buy] = row['Win Rate']
                open_positions = open_positions.append({
                    'Stock A': stock_a_name,
                    'Stock B': stock_b_name,
                    'Entry Signal': signal,
                    'Stock to Buy': stock_to_buy,
                    'Entry Z-Score': latest_z_score,
                    'Half-Life': half_life
                }, ignore_index=True)
        elif signal == 'Exit':
            exits = open_positions[(open_positions['Stock A'] == stock_a_name) & (open_positions['Stock B'] == stock_b_name)]
            if not exits.empty:
                all_signals.append({
                    'Stock A': stock_a_name,
                    'Stock B': stock_b_name,
                    'Signal': signal,
                    'Stock to Sell': stock_to_buy,
                    'Exit Z-Score': latest_z_score,
                    'Half-Life': half_life
                })
                open_positions = open_positions[~((open_positions['Stock A'] == stock_a_name) & (open_positions['Stock B'] == stock_b_name))]

        # Append all buy/exit signals to the list
        if signal != 'Hold':
            all_signals.append({
                'Stock A': stock_a_name,
                'Stock B': stock_b_name,
                'Signal': signal,
                'Stock A Buy/Sell': stock_to_buy if signal == 'Buy' else None,
                'Z-Score': latest_z_score,
                'Half-Life': half_life
            })
    
    # Save all signals to a CSV file
    pd.DataFrame(all_signals).to_csv(output_path, index=False)
    logging.info(f"Signals and half-life saved to {output_path}")


# Main execution
if __name__ == "__main__":
    run_pairs_trading_strategy()