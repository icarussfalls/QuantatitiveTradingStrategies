import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from itertools import combinations
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
config = {
    'risk_free_rate': 0.05,
    'trading_days': 252,
    'commission_rate': 0.004,
    'p_value_threshold': 0.05
}

# Function to process stock data
def process_stock_data(stock_symbols, fill_method='ffill'):
    aligned_data = {}
    for symbol in stock_symbols:
        try:
            data = pd.read_csv(f'{symbol}.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.drop_duplicates(subset='Date')
            data.sort_values(by='Date', ascending=True, inplace=True)
            data.set_index('Date', inplace=True)
            if 'LTP' in data.columns:
                aligned_data[symbol] = data[['LTP']]
            else:
                logging.warning(f"LTP column not found in {symbol}.csv")
        except FileNotFoundError:
            logging.error(f"File {symbol}.csv not found.")
            continue

    # Ensure all data has the same starting date
    if aligned_data:
        max_date = max(df.index.min() for df in aligned_data.values() if not df.empty)
        combined_df = pd.concat([df[df.index >= max_date] for df in aligned_data.values()], axis=1)
        combined_df.columns = [f'{symbol}_LTP' for symbol in aligned_data.keys()]
        combined_df.fillna(method=fill_method, inplace=True)
        return combined_df
    else:
        logging.error("No valid stock data available.")
        return pd.DataFrame()  # Return an empty DataFrame

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate, trading_days=config['trading_days']):
    excess_returns = returns - risk_free_rate / trading_days
    sharpe_ratio = (excess_returns.mean() * trading_days) / excess_returns.std()
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

# Function for Engle-Granger cointegration test
def engle_granger_cointegration(data, p_value_threshold=config['p_value_threshold']):
    stock_symbols = data.columns
    cointegrated_pairs = []
    for pair in combinations(stock_symbols, 2):
        score, p_value, _ = coint(data[pair[0]], data[pair[1]])
        if p_value < p_value_threshold:
            cointegrated_pairs.append(pair)
            logging.info(f"Cointegrated Pair: {pair} with p-value: {p_value}")
    return cointegrated_pairs

# Function for backtesting statistical arbitrage
def backtest_stat_arbitrage(data, stock_a, stock_b, initial_capital=10000, rolling_window=15, 
                            z_entry_threshold=-2.0, z_exit_threshold=0.0, min_holding_period=3, 
                            commission_rate=config['commission_rate'], stop_loss_threshold=0.05):
    spread = data[stock_a] - data[stock_b]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    z_score = (spread - spread_mean) / spread_std
    
    positions = pd.Series(0, index=data.index)
    holding_counter = 0
    in_position = False
    entry_price = 0
    
    for i in range(len(data)):
        if not in_position:
            if z_score.iloc[i] <= z_entry_threshold:
                positions.iloc[i] = 1  # Enter long position on stock_a
                entry_price = spread.iloc[i]  # Record entry price
                in_position = True
                holding_counter = 1
            else:
                positions.iloc[i] = 0
        else:
            holding_counter += 1
            current_loss = (spread.iloc[i] - entry_price) / entry_price
            
            # Check stop-loss condition
            if holding_counter >= min_holding_period and current_loss < -stop_loss_threshold:
                positions.iloc[i] = 0  # Exit position
                in_position = False
                holding_counter = 0
            elif holding_counter >= min_holding_period:
                if z_score.iloc[i] >= z_exit_threshold:
                    positions.iloc[i] = 0  # Exit position
                    in_position = False
                    holding_counter = 0
                else:
                    positions.iloc[i] = 1  # Maintain position
            else:
                positions.iloc[i] = 1  # Maintain position during holding period

    spread_returns = spread.pct_change().fillna(0)
    strategy_returns = positions.shift(1) * spread_returns
    
    # Adjust returns for commissions on trades
    trades = positions.diff().fillna(0)
    commission_costs = trades.abs() * commission_rate
    strategy_returns -= commission_costs.shift(1).fillna(0)

    cumulative_returns = (1 + strategy_returns).cumprod() * initial_capital
    
    if in_position:
        positions.iloc[-1] = 0
        strategy_returns.iloc[-1] = -positions.shift(1).iloc[-1] * spread_returns.iloc[-1]

    max_drawdown, _ = calculate_drawdown(cumulative_returns.pct_change().dropna())
    
    return cumulative_returns, calculate_sharpe_ratio(cumulative_returns.pct_change().dropna(), config['risk_free_rate']), calculate_sortino_ratio(cumulative_returns.pct_change().dropna()), max_drawdown

# Monte Carlo simulation for risk-adjusted returns
def monte_carlo_simulation(stock_a, stock_b, data, num_simulations=1000):
    results = []
    for _ in range(num_simulations):
        rolling_window = np.random.randint(5, 30)
        z_entry_threshold = np.random.uniform(-3.0, -1.0)
        z_exit_threshold = np.random.uniform(0.0, 3.0)

        cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown = backtest_stat_arbitrage(
            data, 
            stock_a,
            stock_b,
            rolling_window=rolling_window,
            z_entry_threshold=z_entry_threshold,
            z_exit_threshold=z_exit_threshold
        )
        
        risk_adjusted_return = sharpe_ratio / (1 + max_drawdown) if max_drawdown != 0 else 0  # Risk-adjusted metric
        results.append({
            'cumulative_returns': cumulative_returns.iloc[-1],
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'risk_adjusted_return': risk_adjusted_return,
            'rolling_window': rolling_window,
            'z_entry_threshold': z_entry_threshold,
            'z_exit_threshold': z_exit_threshold
        })
    
    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    stock_symbols = ['AKPL', 'UPPER', 'NYADI', 'NHPC', 'HDHPC', 'SPDL', 'UNHPL', 'DHPL']
    combined_df = process_stock_data(stock_symbols, fill_method='ffill')

    if not combined_df.empty:
        cointegrated_pairs = engle_granger_cointegration(combined_df)

        all_results = []

        # Optimize parameters for each cointegrated pair
        for stock_a, stock_b in cointegrated_pairs:
            logging.info(f"Running Monte Carlo simulations for pair: {stock_a}, {stock_b}")
            simulations = monte_carlo_simulation(stock_a, stock_b, combined_df)

            # Get the best result based on risk-adjusted return
            best_result = simulations.loc[simulations['risk_adjusted_return'].idxmax()]

            # Backtest with the best parameters to get cumulative returns series
            cumulative_returns, _, _, _ = backtest_stat_arbitrage(
                combined_df,
                stock_a,
                stock_b,
                rolling_window=int(best_result['rolling_window']),
                z_entry_threshold=best_result['z_entry_threshold'],
                z_exit_threshold=best_result['z_exit_threshold']
            )

            all_results.append({
                'Stock A': stock_a,
                'Stock B': stock_b,
                'Best Rolling Window': best_result['rolling_window'],
                'Z Entry Threshold': best_result['z_entry_threshold'],
                'Z Exit Threshold': best_result['z_exit_threshold'],
                'Final Cumulative Returns': best_result['cumulative_returns'],
                'Final Sharpe Ratio': best_result['sharpe_ratio'],
                'Final Sortino Ratio': best_result['sortino_ratio'],
                'Final Max Drawdown': best_result['max_drawdown'],
                'Cumulative Returns Series': cumulative_returns
            })

        results_df = pd.DataFrame(all_results)

        # Save results to CSV
        results_df.to_csv('results.csv', index=False)
        logging.info("Results saved to results.csv")

        # Plot all cumulative returns series
        if not results_df.empty:
            plt.figure(figsize=(12, 6))
            
            for index, row in results_df.iterrows():
                cumulative_returns = row['Cumulative Returns Series']
                plt.plot(cumulative_returns.index, cumulative_returns.values, label=f'{row["Stock A"]} & {row["Stock B"]}')
            
            plt.title('Cumulative Returns of Cointegrated Pairs')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Returns')
            plt.legend()
            plt.grid()
            plt.tight_layout()  # Adjust layout to prevent overlap
            plt.show()

    else:
        logging.error("No valid stock data available for analysis.")
