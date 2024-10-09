import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from itertools import combinations
import os
import logging
from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import minimize
import statsmodels.api as sm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
config = {
    'risk_free_rate': 0.05,
    'trading_days': 252,
    'commission_rate': 0.004,
    'tax_rate': 0.075,
    'p_value_threshold': 0.05,
    'num_simulations': 1000,
    'max_portfolio_size': 10  # Maximum number of pairs in the portfolio
}

# Function to process stock data
def process_stock_data(stock_symbols, fill_method='ffill'):
    aligned_data = {}
    
    for symbol in stock_symbols:
        try:
            data = pd.read_csv(f'/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/{symbol}.csv')
            # Parse the dates and handle invalid dates
            #data['Date'] = pd.to_datetime(data['f_date'], errors='raise')
            data = data.drop_duplicates(subset='Date')
            data.sort_values(by='Date', ascending=True, inplace=True)

            # Set the Date as the index for alignment
            data.set_index('Date', inplace=True)

            # Store the cleaned data
            aligned_data[symbol] = data[['Close']]

        except FileNotFoundError:
            logging.error(f"File {symbol}.csv not found.")
            continue
        except Exception as e:
            logging.error(f"An error occurred while processing {symbol}.csv: {e}")
            continue

    # Ensure all data has the same starting date
    if aligned_data:
        max_date = max(df.index.min() for df in aligned_data.values() if not df.empty)
        combined_df = pd.concat([df[df.index >= max_date] for df in aligned_data.values()], axis=1)
        combined_df.columns = [symbol for symbol in aligned_data.keys()]

        # Drop rows with any missing values
        combined_df.dropna(axis=0, how='any', inplace=True)

        # Drop rows where any LTP is zero
        combined_df = combined_df[(combined_df != 0).all(axis=1)]

        # Save the DataFrame to CSV without extra commas and with handling of NaN
        #combined_df.to_csv('check.csv', index=True)  # Saving with index for dates
        return combined_df
    else:
        logging.error("No valid stock data available.")
        return pd.DataFrame()  # Return an empty DataFrame

def calculate_sharpe_ratio(returns, risk_free_rate=config['risk_free_rate'], trading_days=config['trading_days']):
    excess_returns = returns - (risk_free_rate / trading_days)
    sharpe_ratio = np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate=config['risk_free_rate'], trading_days=config['trading_days']):
    excess_returns = returns - (risk_free_rate / trading_days)
    downside_returns = excess_returns[excess_returns < 0]
    sortino_ratio = np.sqrt(trading_days) * excess_returns.mean() / downside_returns.std()
    return sortino_ratio

def calculate_drawdown(returns):
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (peak - cumulative_returns) / peak
    max_drawdown = drawdown.max()
    return max_drawdown, drawdown

def engle_granger_cointegration(data, p_value_threshold=config['p_value_threshold']):
    stock_symbols = data.columns
    cointegrated_pairs = []
    
    for pair in combinations(stock_symbols, 2):
        score, p_value, _ = coint(data[pair[0]], data[pair[1]])
        if p_value < p_value_threshold:
            cointegrated_pairs.append(pair)
            logging.info(f"Cointegrated Pair: {pair} with p-value: {p_value}")
    
    return cointegrated_pairs

def calculate_spread(data, pair):
    return data[pair[0]] - data[pair[1]]

def calculate_portfolio_value(weights, data):
    return np.dot(data, weights)

def objective_function(weights, data, target_half_life):
    portfolio_value = calculate_portfolio_value(weights, data)
    returns = portfolio_value.pct_change().dropna()
    half_life = calculate_half_life(returns)
    return np.abs(half_life - target_half_life)

def calculate_half_life(returns):
    lag = returns.shift(1)
    ret = returns - lag
    lag = sm.add_constant(lag)
    model = sm.OLS(ret[1:], lag[1:])
    res = model.fit()
    half_life = -np.log(2) / res.params[1]
    return half_life

def optimize_portfolio_weights(data, cointegrated_pairs, target_half_life, sparsity):
    # Flatten the pairs to get unique assets
    unique_assets = list(set([item for pair in cointegrated_pairs for item in pair]))
    n_assets = len(unique_assets)

    initial_weights = np.random.rand(n_assets)
    initial_weights /= np.sum(np.abs(initial_weights))  # Normalize initial weights

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(np.abs(x)) - 1}
    ]
    
    bounds = [(-1, 1) if i < sparsity * n_assets else (0, 0) for i in range(n_assets)]
    
    result = minimize(objective_function, initial_weights, args=(data, target_half_life),
                      method='SLSQP', constraints=constraints, bounds=bounds)
    
    return result.x


def backtest_sparse_mean_reverting(data, cointegrated_pairs, weights, z_entry, z_exit, stop_loss):
    portfolio_values = []
    positions = pd.DataFrame(0, index=data.index, columns=data.columns)
    
    for i, pair in enumerate(cointegrated_pairs):
        if weights[i] != 0:
            spread = calculate_spread(data, pair)
            z_score = (spread - spread.rolling(window=20).mean()) / spread.rolling(window=20).std()
            
            long_entries = (z_score <= -z_entry) & (positions[pair[0]] == 0)
            short_entries = (z_score >= z_entry) & (positions[pair[1]] == 0)
            exits = (z_score.abs() <= z_exit) | (positions[pair[0]].abs() >= stop_loss) | (positions[pair[1]].abs() >= stop_loss)
            
            positions[pair[0]] = np.where(long_entries, 1, np.where(exits, 0, positions[pair[0]]))
            positions[pair[1]] = np.where(short_entries, -1, np.where(exits, 0, positions[pair[1]]))
    
    portfolio_value = (positions * data.pct_change()).sum(axis=1)
    cumulative_returns = (1 + portfolio_value).cumprod()
    
    return cumulative_returns

def run_sparse_mean_reverting_strategy(data, cointegrated_pairs, target_half_life=10, sparsity=0.5, z_entry=2, z_exit=0, stop_loss=0.05):
    weights = optimize_portfolio_weights(data, cointegrated_pairs, target_half_life, sparsity)
    cumulative_returns = backtest_sparse_mean_reverting(data, cointegrated_pairs, weights, z_entry, z_exit, stop_loss)
    
    sharpe_ratio = calculate_sharpe_ratio(cumulative_returns.pct_change().dropna())
    sortino_ratio = calculate_sortino_ratio(cumulative_returns.pct_change().dropna())
    max_drawdown, _ = calculate_drawdown(cumulative_returns.pct_change().dropna())
    
    return cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown, weights

# Stock lists (as in your original code)
stocks_banking = ['NABIL', 'KBL', 'MBL', 'SANIMA', 'NICA']
stocks_finance = ['CFCL', 'GFCL', 'MFIL', 'GUFL']
stocks_microfinance = ['CBBL', 'DDBL', 'SKBBL', 'SMFDB', 'SWBBL', 'SMB', 'FOWAD', 'KLBSL']
stocks_life = ['ALICL', 'LICN', 'NLIC', 'CLI', 'ILI', 'SJLIC']
stocks_non_life = ['NICL', 'NIL', 'NLG', 'SICL', 'PRIN', 'HEI']
stocks_others = ['NRIC', 'NTC', 'SHIVM', 'HRL']
stocks_hydro = ['NYADI', 'RADHI', 'NHPC', 'KPCL', 'HDHPC', 'DHPL', 'API', 'AKPL', 'UNHPL']

stock_categories = {
    'banking': stocks_banking,
    'finance': stocks_finance,
    'microfinance': stocks_microfinance,
    'life': stocks_life,
    'non_life': stocks_non_life,
    'others': stocks_others,
    'hydro': stocks_hydro
}

# Main execution
if __name__ == "__main__":
    all_stocks = [stock for category in stock_categories.values() for stock in category]
    all_stocks = stock_categories['hydro']
    data = process_stock_data(all_stocks)
    print(data)
    
    if not data.empty:
        cointegrated_pairs = engle_granger_cointegration(data)
        
        if len(cointegrated_pairs) > config['max_portfolio_size']:
            cointegrated_pairs = cointegrated_pairs[:config['max_portfolio_size']]
        
        cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown, weights = run_sparse_mean_reverting_strategy(data, cointegrated_pairs)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title('Cumulative Returns of Sparse Mean-Reverting Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('sparse_mean_reverting_portfolio_returns.png')
        plt.close()
        
        # Print results
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print("\nPortfolio Weights:")
        for i, pair in enumerate(cointegrated_pairs):
            if weights[i] != 0:
                print(f"{pair[0]} - {pair[1]}: {weights[i]:.4f}")
    else:
        print("No valid data available for analysis.")