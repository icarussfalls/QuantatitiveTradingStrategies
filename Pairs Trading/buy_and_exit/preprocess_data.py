import pandas as pd
import logging
import os
from statsmodels.tsa.stattools import coint
from itertools import combinations
from statsmodels.regression.linear_model import OLS


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Function to get all filenames from the 'datas' folder
def get_stock_filenames(data_folder='/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/'):
    filenames = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
    stock_symbols = [f.replace('.csv', '') for f in filenames]
    return stock_symbols

# Function to process stock data for a single stock
def process_single_stock(symbol, data_folder='/Users/icarus/Desktop/QuantatitiveTradingStrategies/datas/'):
    try:
        # Read the data from CSV
        data = pd.read_csv(f'{data_folder}/{symbol}.csv')
        
        # Convert 'Date' column to datetime format and handle invalid dates
        data['Date'] = pd.to_datetime(data['date'], errors='coerce')
        data.dropna(subset=['Date'], inplace=True)

        # Remove time part if needed
        if pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = data['Date'].dt.date
        else:
            logging.warning(f"Date column in {symbol} is not datetime-like. Skipping...")
            return None

        # Drop duplicates, sort, and set 'Date' as index
        data = data.drop_duplicates(subset='Date').sort_values(by='Date')
        data.set_index('Date', inplace=True)
        return data[['close']]
    
    except FileNotFoundError:
        logging.error(f"File {symbol}.csv not found.")
    except Exception as e:
        logging.error(f"An error occurred while processing {symbol}.csv: {e}")
    
    return None

# Calculate hedge ratio
def calculate_hedge_ratio(stock_a_prices, stock_b_prices):
    model = OLS(stock_a_prices, stock_b_prices).fit()
    hedge_ratio = model.params[0]
    return hedge_ratio

# Function for Engle-Granger cointegration test with collinearity check
def engle_granger_cointegration(pair_data, p_value_threshold=0.05, correlation_threshold=0.95):
    stock1, stock2 = pair_data.columns
    # Check for high correlation, which can indicate collinearity
    correlation = pair_data[stock1].corr(pair_data[stock2])
    if abs(correlation) > correlation_threshold:
        logging.warning(f"Skipping pair ({stock1}, {stock2}) due to high correlation ({correlation})")
        return None

    # Perform cointegration test
    score, p_value, _ = coint(pair_data[stock1], pair_data[stock2])

    hedge_ratio = calculate_hedge_ratio(pair_data[stock1], pair_data[stock2])
    spread = pair_data[stock1] - hedge_ratio * pair_data[stock2]
    lagged_spread = spread.shift(1)
    spread_diff = spread - lagged_spread
    spread_diff = spread_diff.dropna()
    lagged_spread = lagged_spread.iloc[1:]
    model = OLS(spread_diff, lagged_spread).fit()
    coefficient = model.params[0]
    
    # Exclude non-mean-reverting pairs (p-value >= threshold)
    if p_value < p_value_threshold and coefficient < 0:
        logging.info(f"Cointegrated Pair: ({stock1}, {stock2}) with p-value: {p_value} coefficient {coefficient}")
        return (stock1, stock2, p_value)  # This pair is mean-reverting
    else:
        logging.info(f"Non-cointegrated pair ({stock1}, {stock2}) with p-value: {p_value} coefficient {coefficient}. Excluded.")
        return None

# Main function to execute everything
def main():
    # Get all stock filenames from the datas folder
    stock_symbols = get_stock_filenames()
    results = []  # To store the results (pairs with p-values)

    # Iterate over each unique pair of stocks
    for stock1, stock2 in combinations(stock_symbols, 2):
        # Process data for each stock
        data1 = process_single_stock(stock1)
        data2 = process_single_stock(stock2)

        # If both datasets are valid, align them on the 'Date' index
        if data1 is not None and data2 is not None:
            combined_data = pd.concat([data1, data2], axis=1, join='inner', keys=[stock1, stock2])
            combined_data.columns = [stock1, stock2]

            # Drop any rows with missing data or zeros
            combined_data.dropna(inplace=True)
            combined_data = combined_data[(combined_data != 0).all(axis=1)]

            # Run the Engle-Granger cointegration test
            result = engle_granger_cointegration(combined_data)
            if result:
                results.append(result)  # Store the pair and p-value

    # Save the results to a CSV
    if results:
        results_df = pd.DataFrame(results, columns=['stock_a', 'stock_b', 'P-Value'])
        results_df.to_csv('cointegration_results.csv', index=False)
        logging.info("Cointegration test completed. Results saved to 'cointegration_results.csv'.")
    else:
        logging.info("No cointegrated pairs found.")

# Run the main function
if __name__ == "__main__":
    main()