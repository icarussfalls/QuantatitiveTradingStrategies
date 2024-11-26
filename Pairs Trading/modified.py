import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from itertools import combinations
import os
import logging
from concurrent.futures import ProcessPoolExecutor  # Import for multiprocessing
from statsmodels.regression.linear_model import OLS
from scipy.optimize import minimize
from typing import List, Optional


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
config = {
    'risk_free_rate': 0.05,
    'trading_days': 252,
    'commission_rate': 0.004,
    'tax_rate': 0.075,  # Added tax rate for profitable trades
    'p_value_threshold': 0.05,
    'num_simulations': 10,
    'walk_forward_window': 100,
    'min_window': 5,
    'max_window' : 30
}

# Min and Max window params is for mean reverting process convergence and will determine the rolling window

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

    # Save with both stock names
    output_filename = f'{stock_symbols[0]}_{stock_symbols[1]}_combined_data.csv'
    #combined_df.to_csv(output_filename)
    
    logging.info(f"Processed data for {len(stock_symbols)} stocks")
    return combined_df


def validate_pair(stock_a, stock_b, data, min_periods=30, min_window=config['min_window'], max_window=config['max_window']):
    """
    Validates a pair of stocks for trading based on several criteria including half-life constraints.
    
    Parameters:
    -----------
    stock_a, stock_b : str
        Stock symbols to validate
    data : pd.DataFrame
        DataFrame containing price data for both stocks
    min_periods : int
        Minimum number of periods required for validation
    min_window : int
        Minimum allowed window size for half-life
    max_window : int
        Maximum allowed window size for half-life
        
    Returns:
    --------
    tuple : (is_valid, reason, half_life, rolling_window)
        - is_valid: boolean indicating if pair is valid
        - reason: string explaining validation result
        - half_life: calculated half-life if valid, None if invalid
        - rolling_window: suggested rolling window size if valid, None if invalid
    """
    try:
        # Check for sufficient data
        if len(data) < min_periods:
            return False, "Insufficient data points", None, None

        # Check for missing values
        if data[stock_a].isnull().any() or data[stock_b].isnull().any():
            return False, "Contains missing values", None, None

        # Check for zero or negative prices
        if (data[stock_a] <= 0).any() or (data[stock_b] <= 0).any():
            return False, "Contains zero or negative prices", None, None

        # Calculate spread
        hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])
        spread = data[stock_a] - hedge_ratio * data[stock_b]

        # Calculate half-life
        theta, mu, sigma = OU(spread)
        half_life = half_life_calc(theta)

        # Validate half-life is a valid number
        if half_life <= 0 or np.isnan(half_life) or np.isinf(half_life):
            return False, "Invalid half-life value", None, None

        # Check if half-life is within the allowed window range
        if half_life < min_window:
            return False, f"Half-life ({half_life:.2f}) is less than minimum window ({min_window})", None, None
            
        if half_life > max_window:
            return False, f"Half-life ({half_life:.2f}) exceeds maximum window ({max_window})", None, None

        # Calculate suggested rolling window based on half-life
        rolling_window = int(max(min_window, min(half_life, max_window)))
        
        return True, "Valid pair", half_life, rolling_window

    except Exception as e:
        return False, f"Error during validation: {str(e)}", None, None

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=config['risk_free_rate'], trading_days=config['trading_days']):
    excess_returns = returns - (risk_free_rate / trading_days)
    annualized_excess_returns = excess_returns.mean() * trading_days
    annualized_volatility = excess_returns.std() * np.sqrt(trading_days)
    
    # Avoid division by zero if volatility is zero
    if annualized_volatility == 0:
        return 0

    sharpe_ratio = annualized_excess_returns / annualized_volatility
    return sharpe_ratio


# Function to calculate Sortino ratio
def calculate_sortino_ratio(returns, risk_free_rate=config['risk_free_rate'], trading_days=config['trading_days']):
    downside_returns = returns[returns < risk_free_rate / trading_days]
    expected_return = returns.mean() * trading_days
    downside_deviation = np.sqrt((downside_returns**2).mean()) * trading_days
    return (expected_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0

# Function to calculate drawdown
def calculate_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = drawdown.max()
    return max_drawdown, drawdown



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
        dt = 1  # Assume daily data, so time step is 1 day
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

# Store half-lives for stock pairs
half_life_cache = {}

# Function to calculate half-life only once
def get_half_life(stock_a, stock_b, data, cache=half_life_cache):
    """
    Retrieve the cached half-life for stock pair or calculate if not available.
    """
    pair = tuple(sorted([stock_a, stock_b]))
    if pair in cache:
        return cache[pair]
    else:
        spread = data[stock_a] - calculate_hedge_ratio(data[stock_a], data[stock_b]) * data[stock_b]
        theta, _, _ = OU(spread)
        if not np.isfinite(theta) or theta <= 0:
            logging.warning(f"Skipping pair {stock_a}-{stock_b} due to invalid theta.")
            return None  # Return None for invalid half-life
        half_life = np.log(2) / theta
        cache[pair] = half_life
        return half_life

    
def rolling_mean(arr, window):
    # Compute rolling mean, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].mean() for i in range(len(arr))])

def rolling_std(arr, window):
    # Compute rolling standard deviation, return NaN if there aren't enough data points
    return np.array([np.nan if i < window - 1 else arr[i - window + 1:i + 1].std() for i in range(len(arr))])

    
def calculate_performance_metrics(returns, risk_free_rate=0.03):
    """
    Calculate key performance metrics with enhanced error handling.
    
    Parameters:
    - returns: Strategy returns
    - risk_free_rate: Annual risk-free rate
    
    Returns:
    Dictionary of performance metrics
    """
    # Annualize metrics (assuming daily returns)
    trading_days = 252
    
    # Prevent empty or invalid return series
    if len(returns) == 0:
        return {
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0
        }
    
    # Sharpe Ratio calculation with safety
    excess_returns = returns - (risk_free_rate / trading_days)
    
    # Prevent division by zero and handle small standard deviation
    returns_std = max(excess_returns.std(), 1e-10)
    sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / returns_std
    
    # Sortino Ratio with comprehensive error handling
    downside_returns = returns[returns < 0]
    
    # Calculate Sortino ratio only if there are downside returns
    if len(downside_returns) > 0:
        downside_std = max(downside_returns.std(), 1e-10)
        sortino_ratio = (returns.mean() - (risk_free_rate / trading_days)) / downside_std
    else:
        # If no downside returns, use standard deviation of all returns
        downside_std = max(returns.std(), 1e-10)
        sortino_ratio = (returns.mean() - (risk_free_rate / trading_days)) / downside_std
    
    # Maximum Drawdown calculation
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    
    # Prevent division by zero in drawdown calculation
    drawdown = np.divide(
        cumulative - running_max, 
        running_max, 
        out=np.zeros_like(cumulative), 
        where=running_max!=0
    )
    
    # Handle potential numerical instabilities
    max_drawdown = max(drawdown.min(), -0.99)
    
    return {
        'sharpe_ratio': np.clip(sharpe_ratio, -10, 10),
        'sortino_ratio': np.clip(sortino_ratio, -10, 10),
        'max_drawdown': max_drawdown
    }

def safe_cumulative_returns(strategy_returns, initial_capital, max_drawdown_threshold=0.5):
    """
    Calculate cumulative returns with robust safeguards.
    
    Parameters:
    - strategy_returns: Series of strategy returns
    - initial_capital: Starting capital
    - max_drawdown_threshold: Maximum acceptable percentage drawdown
    
    Returns:
    Pandas Series of cumulative returns with safety checks
    """
    # Prevent extreme outliers and numerical instabilities
    strategy_returns = np.clip(strategy_returns, -0.5, 0.5)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod() * initial_capital
    
    # Track running maximum (peak equity)
    running_max = cumulative_returns.cummax()
    
    # Calculate drawdown
    drawdown = (cumulative_returns - running_max) / running_max
    
    # Apply maximum drawdown constraint
    constrained_returns = cumulative_returns.copy()
    mask = drawdown < -max_drawdown_threshold
    constrained_returns[mask] = running_max[mask]
    
    return constrained_returns

def backtest_stat_arbitrage(data, stock_a, stock_b, initial_capital=10000, 
                            z_entry_threshold_a=-2.0, z_entry_threshold_b=2.0, 
                            z_exit_threshold=0.0, 
                            min_holding_period=3, commission_rate=0.001, 
                            stop_loss_threshold=0.05, cooldown_period=1, 
                            min_window=20, max_window=100, 
                            half_life=None, risk_free_rate=0.03):
    """
    Perform long-only statistical arbitrage backtest with enhanced safety checks.
    
    Parameters:
    - Detailed parameters as in the original function
    
    Returns:
    Tuple of performance metrics and strategy details
    """
    # Validate inputs with more robust checks
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")
    
    if stock_a not in data.columns or stock_b not in data.columns:
        raise ValueError(f"Stocks {stock_a} and {stock_b} must be present in the data")
    
    # Ensure sufficient data
    if len(data) < min_holding_period * 2:
        raise ValueError("Insufficient data for meaningful backtest")
    
    # Equal capital allocation with safety
    capital_a = np.float64(initial_capital / 2)
    capital_b = np.float64(initial_capital / 2)

    # Use provided half-life or calculate it
    if half_life is None:
        half_life = get_half_life(stock_a, stock_b, data)
    
    # Ensure half_life is within reasonable bounds
    half_life = max(min(int(half_life), max_window), min_window)

    # Create a copy of the data to prevent modifications to the original
    data = data.copy()  

    # Calculate returns with safety
    data['returns_a'] = np.clip(data[stock_a].pct_change().fillna(0), -0.5, 0.5)
    data['returns_b'] = np.clip(data[stock_b].pct_change().fillna(0), -0.5, 0.5)
    
    # Hedge ratio calculation with error handling
    try:
        hedge_ratio = calculate_hedge_ratio(data['returns_a'], data['returns_b'])
    except Exception:
        hedge_ratio = 1.0  # Fallback to 1:1 if calculation fails
    
    # Spread calculation
    spread = data['returns_a'] - hedge_ratio * data['returns_b']

    # Calculate z-scores with robust statistics
    spread_mean = spread.rolling(window=int(half_life), min_periods=1).mean()
    spread_std = spread.rolling(window=int(half_life), min_periods=1).std()
    
    # Prevent division by zero and handle edge cases
    spread_std = spread_std.replace(0, 1e-10)
    z_score = (spread - spread_mean) / spread_std

    # Replace infinite values and NaN in z-scores
    z_score = z_score.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Initialize tracking variables
    positions_a = pd.Series(0, index=data.index, dtype=np.float64)
    positions_b = pd.Series(0, index=data.index, dtype=np.float64)

    # Tracking variables with reset to prevent accumulation errors
    holding_counter_a = holding_counter_b = 0
    in_position_a = in_position_b = False
    entry_price_a = entry_price_b = 0
    cooldown_counter_a = cooldown_counter_b = 0

    # Performance tracking with robust initialization
    total_trades = winning_trades = losing_trades = 0
    total_profit = total_loss = 0.0

    # Backtest loop with additional safety checks
    for i in range(1, len(data)):
        # Cooldown management
        cooldown_counter_a = max(0, cooldown_counter_a - 1)
        cooldown_counter_b = max(0, cooldown_counter_b - 1)

        # Entry Conditions for stock_a (long side with separate threshold)
        if not in_position_a and z_score.iloc[i] <= z_entry_threshold_a and cooldown_counter_a == 0:
            share_price_a = data[stock_a].iloc[i]
            position_size_a = max(1, int(capital_a // share_price_a))  # Prevent zero positions
            
            positions_a.iloc[i] = position_size_a  # Long position
            entry_price_a = share_price_a
            in_position_a = True
            holding_counter_a = 1
            
            # Deduct transaction costs on entry
            capital_a -= position_size_a * entry_price_a * (1 + commission_rate)
            total_trades += 1

        # Similar logic for stock_b entry condition (identical to stock_a)
        if not in_position_b and z_score.iloc[i] >= z_entry_threshold_b and cooldown_counter_b == 0:
            share_price_b = data[stock_b].iloc[i]
            position_size_b = max(1, int(capital_b // share_price_b))  # Prevent zero positions
            
            positions_b.iloc[i] = position_size_b  # Long position
            entry_price_b = share_price_b
            in_position_b = True
            holding_counter_b = 1
            
            # Deduct transaction costs on entry
            capital_b -= position_size_b * entry_price_b * (1 + commission_rate)
            total_trades += 1

        # Exit Conditions for stock_a (long position)
        if in_position_a:
            holding_counter_a += 1
            current_gain_a = (data[stock_a].iloc[i] - entry_price_a) / entry_price_a
            
            # Exit conditions
            if (holding_counter_a >= min_holding_period and 
                (current_gain_a < -stop_loss_threshold or z_score.iloc[i] >= z_exit_threshold)):
                
                exit_price_a = data[stock_a].iloc[i]
                trade_profit_a = (exit_price_a - entry_price_a) * positions_a.iloc[i-1]
                
                # Update capital and trade metrics
                capital_a += exit_price_a * positions_a.iloc[i-1] * (1 - commission_rate)
                
                if trade_profit_a > 0:
                    winning_trades += 1
                    total_profit += trade_profit_a
                else:
                    losing_trades += 1
                    total_loss += abs(trade_profit_a)
                
                positions_a.iloc[i] = 0
                in_position_a = False
                cooldown_counter_a = cooldown_period

        # Identical exit conditions for stock_b
        if in_position_b:
            holding_counter_b += 1
            current_gain_b = (data[stock_b].iloc[i] - entry_price_b) / entry_price_b
            
            # Exit conditions
            if (holding_counter_b >= min_holding_period and 
                (current_gain_b < -stop_loss_threshold or z_score.iloc[i] >= z_exit_threshold)):
                
                exit_price_b = data[stock_b].iloc[i]
                trade_profit_b = (exit_price_b - entry_price_b) * positions_b.iloc[i-1]
                
                # Update capital and trade metrics
                capital_b += exit_price_b * positions_b.iloc[i-1] * (1 - commission_rate)
                
                if trade_profit_b > 0:
                    winning_trades += 1
                    total_profit += trade_profit_b
                else:
                    losing_trades += 1
                    total_loss += abs(trade_profit_b)
                
                positions_b.iloc[i] = 0
                in_position_b = False
                cooldown_counter_b = cooldown_period

    # Final capital calculation with sanity check
    final_capital = min(max(capital_a + capital_b, initial_capital * 0.5), initial_capital * 2)

    # Performance metrics calculation
    stock_returns_a = np.clip(data[stock_a].pct_change().fillna(0), -0.5, 0.5)
    stock_returns_b = np.clip(data[stock_b].pct_change().fillna(0), -0.5, 0.5)
    strategy_returns_a = (positions_a.shift(1) * stock_returns_a).fillna(0)
    strategy_returns_b = (positions_b.shift(1) * stock_returns_b).fillna(0)
    strategy_returns = strategy_returns_a + strategy_returns_b
    
    # Trades and commission costs
    trades_a = positions_a.diff().fillna(0)
    trades_b = positions_b.diff().fillna(0)
    commission_costs = (trades_a.abs() + trades_b.abs()) * commission_rate
    strategy_returns -= commission_costs.shift(1).fillna(0)
    
    # Safe cumulative returns calculation
    cumulative_returns = safe_cumulative_returns(strategy_returns, initial_capital)
    
    # Performance metrics
    performance_metrics = calculate_performance_metrics(strategy_returns, risk_free_rate)

    # Additional metrics with safety checks
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = total_profit / (total_loss + 1e-10)
    average_profit = total_profit / (winning_trades + 1e-10)
    average_loss = total_loss / (losing_trades + 1e-10)

    return (cumulative_returns, 
            performance_metrics['sharpe_ratio'], 
            performance_metrics['sortino_ratio'], 
            performance_metrics['max_drawdown'], 
            total_trades, 
            win_rate, 
            profit_factor, 
            average_profit, 
            average_loss, 
            hedge_ratio, 
            half_life)

def monte_carlo_simulation(stock_a, stock_b, data, num_simulations=config['num_simulations'], 
                         walk_forward_window=config['walk_forward_window'],
                         min_window=config['min_window'], max_window=config['max_window']):
    """
    Modified Monte Carlo simulation
    """
    
    # Proceed with simulation only for valid pairs
    results = []
    for sim in range(num_simulations):
        # Randomize parameters for each simulation
        z_entry_threshold_a = np.random.uniform(-3.0, -1.0)  # For stock_a (long)
        z_entry_threshold_b = np.random.uniform(1.0, 3.0)    # For stock_b (long)
        z_exit_threshold = np.random.uniform(-0.5, 0.5)      # Exit threshold
        
        # Iterate over different stop-loss thresholds
        for stop_loss_threshold in np.arange(0.05, 0.21, 0.05):
            # Divide data into training and testing sets
            train_start = 0
            train_end = len(data) - walk_forward_window
            test_start = train_end
            test_end = len(data)

            while test_end <= len(data):
                train_data = data.iloc[train_start:train_end]
                test_data = data.iloc[test_start:test_end]

                # Backtest on training data
                (cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown,
                 total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio, half_life) = backtest_stat_arbitrage(
                    train_data,
                    stock_a,
                    stock_b,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    min_window=min_window,
                    max_window=max_window
                )


                # Evaluate on testing data
                (test_cumulative_returns, _, _, _, _, _, _, _, _, _, _) = backtest_stat_arbitrage(
                    test_data,
                    stock_a,
                    stock_b,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold,
                    min_window=min_window,
                    max_window=max_window
                )

                epsilon = 1e-6  # Small value to avoid division by zero
                risk_adjusted_return = sharpe_ratio / (1 + max_drawdown + epsilon)


                results.append({
                    'cumulative_returns': cumulative_returns.iloc[-1],
                    'test_cumulative_returns': test_cumulative_returns.iloc[-1],
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'risk_adjusted_return': risk_adjusted_return,
                    'z_entry_threshold_a': z_entry_threshold_a,
                    'z_entry_threshold_b': z_entry_threshold_b,
                    'z_exit_threshold': z_exit_threshold,
                    'stop_loss_threshold': stop_loss_threshold,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_profit': average_profit,
                    'average_loss': average_loss,
                    'hedge_ratio': hedge_ratio,
                    'half_life': half_life,  # Add this line to store the rolling window
                })

                # Update training and testing windows
                train_start += walk_forward_window
                train_end += walk_forward_window
                test_start += walk_forward_window
                test_end += walk_forward_window

    return pd.DataFrame(results)



# Getting the stock list from the datas folder
# Function to get all filenames from the 'datas' folder
def get_stock_filenames(data_folder='/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/'):
    filenames = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    stock_symbols = [f.replace('.csv', '') for f in filenames]
    return stock_symbols

# Cointegrated assets
# Load cointegrated pairs
def load_cointegrated_pairs(filepath='/Users/icarus/Desktop/QuantatitiveTradingStrategies/cointegration_results.csv'):
    return pd.read_csv(filepath)

# Change the current working directory
os.chdir('/Users/icarus/Desktop/QuantatitiveTradingStrategies/Pairs Trading')

# Create directories for results and plots
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Check for existing validation results
VALIDATION_RESULTS_PATH = 'results/pair_validation_results.csv'

if __name__ == "__main__":
    all_final_results = []

    # Load cointegrated pairs from file
    cointegrated_pairs = load_cointegrated_pairs()

    # Check if validation results already exist
    if os.path.exists(VALIDATION_RESULTS_PATH):
        logging.info("Loading existing validation results.")
        validation_results = pd.read_csv(VALIDATION_RESULTS_PATH)

        # Filter valid and rejected pairs
        valid_pairs = validation_results[validation_results['status'] == 'valid'][['stock_a', 'stock_b', 'reason/half_life']].values.tolist()
        rejected_pairs = validation_results[validation_results['status'] == 'rejected'][['stock_a', 'stock_b', 'reason/half_life']].values.tolist()

    else:
        logging.info("Validation results not found. Starting validation process.")
        valid_pairs = []
        rejected_pairs = []

        # First pass: validate all pairs
        for _, row in cointegrated_pairs.iterrows():
            stock_a, stock_b = row['stock_a'], row['stock_b']
            data = process_stock_data([stock_a, stock_b], fill_method='ffill')

            is_valid, reason, half_life, rolling_window = validate_pair(stock_a, stock_b, data)

            # Check if half_life is valid (not None) and meets conditions
            if is_valid and half_life and rolling_window is not None:
                rolling_window = int(max(1, min(max(half_life, 10), 50)))

                if rolling_window < 1:
                    # Exclude pair if rolling_window is invalid
                    rejected_pairs.append((stock_a, stock_b, "Invalid rolling window"))
                    logging.warning(f"Rejected pair: {stock_a}-{stock_b} (Invalid rolling window)")
                    continue

                valid_pairs.append((stock_a, stock_b, half_life))
                logging.info(f"Validated pair: {stock_a}-{stock_b} (Half-life: {half_life})")
            else:
                # Exclude pair if half_life is None or other validation issues
                reason = reason or "None half-life"
                rejected_pairs.append((stock_a, stock_b, reason))
                logging.warning(f"Rejected pair: {stock_a}-{stock_b} ({reason})")

        # Save validation results to file
        validation_results = pd.DataFrame({
            'stock_a': [p[0] for p in valid_pairs + rejected_pairs],
            'stock_b': [p[1] for p in valid_pairs + rejected_pairs],
            'status': ['valid' for _ in valid_pairs] + ['rejected' for _ in rejected_pairs],
            'reason/half_life': [p[2] for p in valid_pairs + rejected_pairs]
        })
        os.makedirs(os.path.dirname(VALIDATION_RESULTS_PATH), exist_ok=True)
        validation_results.to_csv(VALIDATION_RESULTS_PATH, index=False)
        logging.info(f"Validation results saved to {VALIDATION_RESULTS_PATH}.")

    # Proceed with analysis only for valid pairs
    if valid_pairs:
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(
                    monte_carlo_simulation, 
                    pair[0], pair[1], 
                    process_stock_data([pair[0], pair[1]], fill_method='ffill')
                ): pair for pair in valid_pairs
            }

            for future in futures:
                stock_a, stock_b, half_life = futures[future]
                logging.info(f"Running Monte Carlo simulations for pair: {stock_a}, {stock_b}")

                simulations = future.result()

                if simulations.empty:
                    logging.warning(f"No simulation results for pair: {stock_a}, {stock_b}")
                    continue

                best_result = simulations.loc[simulations['risk_adjusted_return'].idxmax()]

                # Run backtest with best parameters using combined_df for the current pair
                cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown, total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio, half_life = backtest_stat_arbitrage(
                    data=process_stock_data([stock_a, stock_b], fill_method='ffill'),  # Process data for this pair
                    stock_a=stock_a,
                    stock_b=stock_b,
                    z_entry_threshold_a=best_result['z_entry_threshold_a'],
                    z_entry_threshold_b=best_result['z_entry_threshold_b'],
                    z_exit_threshold=best_result['z_exit_threshold'],
                    stop_loss_threshold=best_result['stop_loss_threshold']
                )

                all_final_results.append({
                    'Stock A': stock_a,
                    'Stock B': stock_b,
                    'Z Entry Threshold A': best_result['z_entry_threshold_a'],
                    'Z Entry Threshold B': best_result['z_entry_threshold_b'],
                    'Z Exit Threshold': best_result['z_exit_threshold'],
                    'Stop Loss Threshold': best_result['stop_loss_threshold'],
                    'Final Cumulative Returns': cumulative_returns.iloc[-1],
                    'Final Sharpe Ratio': sharpe_ratio,
                    'Final Sortino Ratio': sortino_ratio,
                    'Final Max Drawdown': max_drawdown,
                    'Cumulative Returns Series': cumulative_returns.values.tolist(),
                    'Total Trades': total_trades,
                    'Win Rate': win_rate,
                    'Profit Factor': profit_factor,
                    'Average Profit': average_profit,
                    'Average Loss': average_loss,
                    'Hedge Ratio': hedge_ratio,
                    'Half Life': half_life
                })


    # Save and plot results as before
    if all_final_results:
        results_df = pd.DataFrame(all_final_results)  # Convert list of dicts to DataFrame

        if 'Cumulative Returns Series' in results_df.columns:
            cumulative_returns_df = pd.DataFrame(results_df['Cumulative Returns Series'].tolist())
            cumulative_returns_df.columns = [f'Timestep {i+1}' for i in range(cumulative_returns_df.shape[1])]

            final_results_df = pd.concat([results_df.drop(columns='Cumulative Returns Series'), cumulative_returns_df], axis=1)

            # Save results to CSV
            csv_filename = 'results/cointegrated_pairs_results.csv'
            final_results_df.to_csv(csv_filename, index=False)
            logging.info(f"Results saved to {csv_filename}")

            # Plot cumulative returns series
            plt.figure(figsize=(12, 6))
            for index, row in final_results_df.iterrows():
                cumulative_returns_series = row[['Timestep 1'] + [f'Timestep {i+1}' for i in range(1, cumulative_returns_df.shape[1])]].values
                cumulative_returns_series = [float(value) for value in cumulative_returns_series]

                # Ensure the date index is in datetime format
                date_index = pd.to_datetime(process_stock_data([row['Stock A'], row['Stock B']], fill_method='ffill').index[:len(cumulative_returns_series)])

                plt.plot(date_index, cumulative_returns_series, label=f'{row["Stock A"]} & {row["Stock B"]}')

            plt.title('Cumulative Returns of Cointegrated Pairs')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plot_filename = 'plots/cumulative_returns_cointegrated_pairs.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logging.info(f"Cumulative returns plot saved to {plot_filename}")
            plt.close()

        else:
            logging.warning("'Cumulative Returns Series' not found in results.")
    else:
        logging.error("No valid stock data available for analysis.")
