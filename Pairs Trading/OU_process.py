import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.regression.linear_model import OLS
import logging
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

def process_stock_data(stock_symbols, fill_method='ffill'):
    aligned_data = {}
    
    for symbol in stock_symbols:
        try:
            data = pd.read_csv(f'/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/{symbol}.csv')
            # Parse the dates and set the Date as the index
            data['Date'] = pd.to_datetime(data['date'], errors='coerce')  # Handle invalid dates
            #data.set_index('Date', inplace=True)

            # Store the cleaned data (only the 'close' price)
            aligned_data[symbol] = data[['close']]

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

        # Fill missing values and drop rows with zeros
        combined_df.fillna(method=fill_method, inplace=True)
        combined_df = combined_df[(combined_df != 0).all(axis=1)]

        return combined_df
    else:
        logging.error("No valid stock data available.")
        return pd.DataFrame()  # Return an empty DataFrame

def OU(spread):
    """
    Fit an Ornstein-Uhlenbeck process to the given spread data.
    Returns estimated parameters: theta, mu, sigma.
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


# Read the CSV file with pair validation results
pairs = pd.read_csv("Pairs Trading/results/pair_validation_results.csv")

# Strip spaces from column names if necessary
pairs.columns = pairs.columns.str.strip()

# Drop rows with NaN in 'status' and filter for 'Valid' pairs
pairs = pairs[pairs['status'].str.lower() == 'valid']

def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio

# Define stock symbols and read their data
stock_a = 'UNHPL'
stock_b = 'KSY'

# Use the process_stock_data function to align data for both stocks
data = process_stock_data([stock_a, stock_b], fill_method='ffill')

# Check if the columns exist in the 'data' DataFrame
if stock_a in data.columns and stock_b in data.columns:
    # Calculate the hedge ratio between the two stocks
    hedge_ratio = calculate_hedge_ratio(data[stock_a], data[stock_b])

    # Calculate the spread using the hedge ratio
    spread = data[stock_a] - hedge_ratio * data[stock_b]

    # Fit the Ornstein-Uhlenbeck process to the spread
    theta, mu, sigma = OU(spread)

    # Calculate the half-life based on the fitted theta
    half_life = np.log(2) / theta

    # Print the half-life
    print(f"Half-life: {half_life:.2f} days")


    # Print the results
    print(f"theta: {theta}, mu: {mu}, sigma: {sigma}")
    
    # Plot the spread
    plt.figure(figsize=(10, 6))
    plt.plot(spread.index, spread.values, label='Spread', color='blue')
    plt.axhline(y=mu, color='red', linestyle='--', label=f'Mean (mu) = {mu:.2f}')
    plt.title(f"Spread between {stock_a} and {stock_b}")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    logging.error(f"Columns '{stock_a}' and '{stock_b}' not found in the data.")
