import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from itertools import combinations
import os
import logging
from concurrent.futures import ProcessPoolExecutor  # Import for multiprocessing
from statsmodels.regression.linear_model import OLS


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration dictionary
config = {
    'risk_free_rate': 0.05,
    'trading_days': 252,
    'commission_rate': 0.004,
    'tax_rate': 0.075,  # Added tax rate for profitable trades
    'p_value_threshold': 0.05,
    'num_simulations': 1000,
    'walk_forward_window': 20
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

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate, trading_days=config['trading_days']):
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

# Function for Engle-Granger cointegration test
def engle_granger_cointegration(data, p_value_threshold=config['p_value_threshold']):
    stock_symbols = data.columns
    cointegrated_pairs = set()  # Use a set to avoid duplicates
    
    for pair in combinations(stock_symbols, 2):
        sorted_pair = tuple(sorted(pair))  # Ensure pairs are always in the same order
        if sorted_pair not in cointegrated_pairs:  # Only process if not already done
            score, p_value, _ = coint(data[pair[0]], data[pair[1]])
            if p_value < p_value_threshold:
                cointegrated_pairs.add(sorted_pair)  # Add the pair in sorted order
                logging.info(f"Cointegrated Pair: {pair} with p-value: {p_value}")
    
    return list(cointegrated_pairs)

# Function to calculate hedge ratio
def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio

# Function to backtest the strategy
def backtest_stat_arbitrage(data, stock_a, stock_b, initial_capital=10000, rolling_window=15, 
                            z_entry_threshold_a=-2.0, z_entry_threshold_b=2.0, z_exit_threshold=0.0, 
                            min_holding_period=3, commission_rate=0.001, stop_loss_threshold=0.05, 
                            cooldown_period=1):  # Add cooldown period parameter
    
    hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])
    spread = data[stock_a] - hedge_ratio * data[stock_b]
    spread_mean = spread.rolling(window=rolling_window).mean()
    spread_std = spread.rolling(window=rolling_window).std()
    z_score = (spread - spread_mean) / spread_std
    
    positions_a = pd.Series(0, index=data.index)
    positions_b = pd.Series(0, index=data.index)
    capital = initial_capital
    
    holding_counter_a = 0
    holding_counter_b = 0
    in_position_a = False
    in_position_b = False
    entry_price_a = 0
    entry_price_b = 0
    
    # Cooldown counters
    cooldown_counter_a = 0
    cooldown_counter_b = 0
    
    # Trade metrics
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    total_profit = 0
    total_loss = 0
    
    for i in range(1, len(data)):
        # --- Cooldown Logic ---
        if cooldown_counter_a > 0:
            cooldown_counter_a -= 1
        if cooldown_counter_b > 0:
            cooldown_counter_b -= 1
        
        # --- Entry Conditions ---
        if not in_position_a and z_score.iloc[i] <= z_entry_threshold_a and cooldown_counter_a == 0:
            positions_a.iloc[i] = 1  # Go long on stock_a
            entry_price_a = data[stock_a].iloc[i]  # Record the entry price of stock_a
            in_position_a = True
            holding_counter_a = 1
            capital -= entry_price_a * (1 + commission_rate)  # Adjust capital for commission
            total_trades += 1
            
        if not in_position_b and z_score.iloc[i] >= z_entry_threshold_b and cooldown_counter_b == 0:
            positions_b.iloc[i] = 1  # Go long on stock_b
            entry_price_b = data[stock_b].iloc[i]  # Record the entry price of stock_b
            in_position_b = True
            holding_counter_b = 1
            capital -= entry_price_b * (1 + commission_rate)  # Adjust capital for commission
            total_trades += 1
        
        # --- Exit for stock_a ---
        if in_position_a:
            holding_counter_a += 1
            current_loss_a = (data[stock_a].iloc[i] - entry_price_a) / entry_price_a
            
            # Exit on stop-loss or if z-score has reverted beyond exit threshold for stock_a
            if (holding_counter_a >= min_holding_period and current_loss_a < -stop_loss_threshold) or \
               (holding_counter_a >= min_holding_period and z_score.iloc[i] >= z_exit_threshold):
                positions_a.iloc[i] = -positions_a.iloc[i-1]  # Exit position for stock_a
                exit_price_a = data[stock_a].iloc[i]
                trade_profit_a = (exit_price_a - entry_price_a)  # Calculate profit without shares
                if trade_profit_a > 0:
                    winning_trades += 1
                    trade_profit_a_after_tax = trade_profit_a * (1 - config['tax_rate'])  # Apply tax
                    total_profit += trade_profit_a_after_tax
                else:
                    losing_trades += 1
                    total_loss += abs(trade_profit_a)
                in_position_a = False
                holding_counter_a = 0
                cooldown_counter_a = cooldown_period  # Reset cooldown for stock_a
                capital += exit_price_a * (1 - commission_rate)  # Adjust capital for commission
        
        # --- Exit for stock_b ---
        if in_position_b:
            holding_counter_b += 1
            current_loss_b = (data[stock_b].iloc[i] - entry_price_b) / entry_price_b
            
            # Exit on stop-loss or if z-score has reverted beyond exit threshold for stock_b
            if (holding_counter_b >= min_holding_period and current_loss_b < -stop_loss_threshold) or \
               (holding_counter_b >= min_holding_period and z_score.iloc[i] <= z_exit_threshold):
                positions_b.iloc[i] = -positions_b.iloc[i-1]  # Exit position for stock_b
                exit_price_b = data[stock_b].iloc[i]
                trade_profit_b = (exit_price_b - entry_price_b)  # Calculate profit without shares
                if trade_profit_b > 0:
                    winning_trades += 1
                    trade_profit_b_after_tax = trade_profit_b * (1 - config['tax_rate'])  # Apply tax
                    total_profit += trade_profit_b_after_tax
                else:
                    losing_trades += 1
                    total_loss += abs(trade_profit_b)
                in_position_b = False
                holding_counter_b = 0
                cooldown_counter_b = cooldown_period  # Reset cooldown for stock_b
                capital += exit_price_b * (1 - commission_rate)  # Adjust capital for commission
    
    # Calculate returns for stock_a and stock_b
    stock_returns_a = data[stock_a].pct_change().fillna(0)
    stock_returns_b = data[stock_b].pct_change().fillna(0)
    
    # Combine strategy returns from both stocks
    strategy_returns_a = (positions_a.shift(1) * stock_returns_a).fillna(0)
    strategy_returns_b = (positions_b.shift(1) * stock_returns_b).fillna(0)
    strategy_returns = strategy_returns_a + strategy_returns_b
    
    # Adjust returns for commissions on trades
    trades_a = positions_a.diff().fillna(0)
    trades_b = positions_b.diff().fillna(0)
    commission_costs = (trades_a.abs() + trades_b.abs()) * commission_rate
    strategy_returns -= commission_costs.shift(1).fillna(0)
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod() * initial_capital
    
    # Ensure we exit positions on the last day if still in position
    if in_position_a:
        positions_a.iloc[-1] = -positions_a.shift(1).iloc[-1]  # Exit position for stock_a
        strategy_returns_a.iloc[-1] = -positions_a.shift(1).iloc[-1] * stock_returns_a.iloc[-1]
    if in_position_b:
        positions_b.iloc[-1] = -positions_b.shift(1).iloc[-1]  # Exit position for stock_b
        strategy_returns_b.iloc[-1] = -positions_b.shift(1).iloc[-1] * stock_returns_b.iloc[-1]
    
    # Recalculate combined strategy returns after exiting
    strategy_returns = strategy_returns_a + strategy_returns_b
    
    # Calculate performance metrics: drawdown, Sharpe ratio, Sortino ratio
    max_drawdown, _ = calculate_drawdown(cumulative_returns.pct_change().dropna())
    sharpe_ratio = calculate_sharpe_ratio(cumulative_returns.pct_change().dropna(), config['risk_free_rate'])
    sortino_ratio = calculate_sortino_ratio(cumulative_returns.pct_change().dropna())
    
    # Calculate additional trade metrics
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    average_profit = total_profit / winning_trades if winning_trades > 0 else 0
    average_loss = total_loss / losing_trades if losing_trades > 0 else 0
    
    return (cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown, 
            total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio)



# Monte Carlo simulation for risk-adjusted returns with stop-loss optimization
def monte_carlo_simulation(stock_a, stock_b, data, num_simulations=config['num_simulations'], walk_forward_window=config['walk_forward_window']):
    results = []
    for sim in range(num_simulations):
        # Randomize parameters for each simulation
        rolling_window = np.random.randint(5, 30)
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
                 total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio) = backtest_stat_arbitrage(
                    train_data,
                    stock_a,
                    stock_b,
                    rolling_window=rolling_window,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold
                )

                # Evaluate on testing data
                (test_cumulative_returns, _, _, _, _, _, _, _, _, _) = backtest_stat_arbitrage(
                    test_data,
                    stock_a,
                    stock_b,
                    rolling_window=rolling_window,
                    z_entry_threshold_a=z_entry_threshold_a,
                    z_entry_threshold_b=z_entry_threshold_b,
                    z_exit_threshold=z_exit_threshold,
                    stop_loss_threshold=stop_loss_threshold
                )

                risk_adjusted_return = sharpe_ratio / (1 + max_drawdown) if max_drawdown != 0 else 0

                results.append({
                    'cumulative_returns': cumulative_returns.iloc[-1],
                    'test_cumulative_returns': test_cumulative_returns.iloc[-1],
                    'sharpe_ratio': sharpe_ratio,
                    'sortino_ratio': sortino_ratio,
                    'max_drawdown': max_drawdown,
                    'risk_adjusted_return': risk_adjusted_return,
                    'rolling_window': rolling_window,
                    'z_entry_threshold_a': z_entry_threshold_a,
                    'z_entry_threshold_b': z_entry_threshold_b,
                    'z_exit_threshold': z_exit_threshold,
                    'stop_loss_threshold': stop_loss_threshold,
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_profit': average_profit,
                    'average_loss': average_loss,
                    'hedge_ratio': hedge_ratio
                })

                # Update training and testing windows
                train_start += walk_forward_window
                train_end += walk_forward_window
                test_start += walk_forward_window
                test_end += walk_forward_window

    return pd.DataFrame(results)

# Stock lists
stocks_banking = ['NABIL', 'KBL', 'MBL', 'SANIMA', 'NICA']
stocks_finance = ['CFCL', 'GFCL', 'MFIL', 'GUFL', 'NFL']
stocks_microfinance = ['CBBL', 'DDBL', 'SKBBL', 'SMFDB', 'SWBBL', 'SMB', 'FOWAD', 'KLBSL']
stocks_life = ['ALICL', 'LICN', 'NLIC', 'CLI', 'ILI', 'SJLIC']
stocks_non_life = ['NICL', 'NIL', 'NLG', 'SICL', 'PRIN', 'HEI']
stocks_others = ['NRIC', 'NTC', 'SHIVM', 'HRL']
stocks_hydro = ['NYADI', 'RADHI', 'NHPC', 'KPCL', 'HDHPC', 'DHPL', 'API', 'AKPL', 'UNHPL']

# Define stock category names and their lists
stock_categories = {
    'banking': stocks_banking,
    'finance': stocks_finance,
    'microfinance': stocks_microfinance,
    'life': stocks_life,
    'non_life': stocks_non_life,
    'others': stocks_others,
    'hydro': stocks_hydro
}

# Change the current working directory
os.chdir('/Users/icarus/Desktop/QuantatitiveTradingStrategies/Pairs Trading')

# Create directories for results and plots
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Main execution
if __name__ == "__main__":
    all_final_results = pd.DataFrame()

    # Generate combinations of stock categories (pairwise)
    category_combinations = list(combinations(stock_categories.items(), 2))

    for (category_1, stock_list_1), (category_2, stock_list_2) in category_combinations:
        combined_category = f"{category_1}_and_{category_2}"
        logging.info(f"Processing cross category: {combined_category}")

        # Combine the two stock lists
        stock_symbols = stock_list_1 + stock_list_2

        # Process stock data for the combined category
        combined_df = process_stock_data(stock_symbols, fill_method='ffill')

        if not combined_df.empty:
            cointegrated_pairs = engle_granger_cointegration(combined_df)

            all_results = []

            # Use ProcessPoolExecutor for parallel processing
            with ProcessPoolExecutor() as executor:
                # Submit Monte Carlo simulations for each cointegrated pair
                futures = {
                    executor.submit(monte_carlo_simulation, stock_a, stock_b, combined_df): (stock_a, stock_b)
                    for stock_a, stock_b in cointegrated_pairs
                }

                for future in futures:
                    stock_a, stock_b = futures[future]
                    logging.info(f"Running Monte Carlo simulations for pair: {stock_a}, {stock_b}")
                    simulations = future.result()

                    if simulations.empty:
                        logging.warning(f"No simulation results for pair: {stock_a}, {stock_b}")
                        continue

                    best_result = simulations.loc[simulations['risk_adjusted_return'].idxmax()]

                    (cumulative_returns, sharpe_ratio, sortino_ratio, max_drawdown,
                    total_trades, win_rate, profit_factor, average_profit, average_loss, hedge_ratio) = backtest_stat_arbitrage(
                        combined_df,
                        stock_a,
                        stock_b,
                        rolling_window=int(best_result['rolling_window']),
                        z_entry_threshold_a=best_result['z_entry_threshold_a'],
                        z_entry_threshold_b=best_result['z_entry_threshold_b'],
                        z_exit_threshold=best_result['z_exit_threshold'],
                        stop_loss_threshold=best_result['stop_loss_threshold']
                    )

                    if cumulative_returns is None or cumulative_returns.empty:
                        logging.warning(f"No cumulative returns calculated for pair: {stock_a}, {stock_b}")
                        continue

                    all_results.append({
                        'Stock A': stock_a,
                        'Stock B': stock_b,
                        'Best Rolling Window': best_result['rolling_window'],
                        'Z Entry Threshold A': best_result['z_entry_threshold_a'],
                        'Z Entry Threshold B': best_result['z_entry_threshold_b'],
                        'Z Exit Threshold': best_result['z_exit_threshold'],
                        'Stop Loss Threshold': best_result['stop_loss_threshold'],
                        'Final Cumulative Returns': cumulative_returns.iloc[-1],  # Final value of cumulative returns
                        'Final Sharpe Ratio': sharpe_ratio,
                        'Final Sortino Ratio': sortino_ratio,
                        'Final Max Drawdown': max_drawdown,
                        'Cumulative Returns Series': cumulative_returns.values.tolist(),  # Store the entire series
                        'Total Trades': total_trades,
                        'Win Rate': win_rate,
                        'Profit Factor': profit_factor,
                        'Average Profit': average_profit,
                        'Average Loss': average_loss,
                        'Hedge Ratio': hedge_ratio
                    })

            # Check if there are any results
            if all_results:
                results_df = pd.DataFrame(all_results)

                # Extract the cumulative returns series and ensure it exists
                if 'Cumulative Returns Series' in results_df.columns:
                    cumulative_returns_df = pd.DataFrame(results_df['Cumulative Returns Series'].tolist())
                    cumulative_returns_df.columns = [f'Timestep {i+1}' for i in range(cumulative_returns_df.shape[1])]

                    # Concatenate with results_df to maintain structure
                    final_results_df = pd.concat([results_df.drop(columns='Cumulative Returns Series'), cumulative_returns_df], axis=1)


                    # Save results to CSV in the results directory
                    csv_filename = f'results/{combined_category}.csv'
                    final_results_df.to_csv(csv_filename, index=False)
                    logging.info(f"Results for {combined_category} saved to {csv_filename}")

                    # Append current results to the main dataframe that holds all results
                    all_final_results = pd.concat([all_final_results, final_results_df])

                    

                    # Plot cumulative returns series
                    plt.figure(figsize=(12, 6))

                    for index, row in final_results_df.iterrows():
                        # Convert cumulative returns series to float
                        cumulative_returns_series = row[['Timestep 1'] + [f'Timestep {i+1}' for i in range(1, cumulative_returns_df.shape[1])]].values
                        cumulative_returns_series = [float(value) for value in cumulative_returns_series]  # Ensure the values are floats

                        # Ensure the date index is in datetime format
                        date_index = pd.to_datetime(combined_df.index[:len(cumulative_returns_series)])

                        plt.plot(date_index, cumulative_returns_series, label=f'{row["Stock A"]} & {row["Stock B"]}')

                    plt.title(f'Cumulative Returns of Cointegrated Pairs ({combined_category})')
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative Returns')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()

                    # Save the plot in the plots directory
                    plot_filename = f'plots/cumulative_returns_{combined_category}.png'
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    logging.info(f"Cumulative returns plot saved to {plot_filename}")
                    plt.close()  # Close the plot to avoid displaying it
                else:
                    logging.warning(f"'Cumulative Returns Series' not found in results for {combined_category}")
            else:
                logging.error(f"No valid stock data available for {combined_category} analysis.")

