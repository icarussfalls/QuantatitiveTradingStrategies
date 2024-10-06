import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

def log_likelihood(params, X, dt):
    """Log-likelihood function for OU process."""
    theta, mu, sigma = params
    N = len(X)
    sum_X = np.sum(X[:-1])
    sum_Y = np.sum(X[1:])
    sum_XX = np.sum(X[:-1]**2)
    sum_XY = np.sum(X[:-1] * X[1:])
    
    S1 = sum_Y - sum_X * np.exp(-theta * dt)
    S2 = sum_XX + sum_Y**2 * np.exp(-2*theta*dt) - 2*np.exp(-theta*dt) * sum_XY
    
    mu_hat = S1 / (N * (1 - np.exp(-theta * dt)))
    sigma_squared_hat = S2 * 2 * theta / (N * (1 - np.exp(-2 * theta * dt)))
    
    log_lik = -N/2 * np.log(2*np.pi*sigma_squared_hat) - theta*S2/(sigma_squared_hat * (1 - np.exp(-2*theta*dt)))
    return -log_lik

def estimate_params(X, dt):
    """Estimate OU process parameters using MLE."""
    initial_guess = [0.1, np.mean(X), np.std(X)]
    bounds = ((1e-5, None), (None, None), (1e-5, None))
    result = minimize(log_likelihood, initial_guess, args=(X, dt), bounds=bounds)
    return result.x

def trading_signals(X, mu, threshold):
    """Generate trading signals based on OU process."""
    signals = np.zeros_like(X)
    signals[X < mu - threshold] = 1  # Buy signal
    signals[X > mu + threshold] = -1  # Sell signal
    return signals

def backtest_strategy(X, signals):
    """Backtest the trading strategy."""
    returns = np.diff(X) / X[:-1]  # Use percentage returns
    strategy_returns = signals[:-1] * returns
    cumulative_returns = np.cumprod(1 + strategy_returns) - 1
    return cumulative_returns

# Fetch intraday stock data
ticker = "AAPL"  # Apple Inc. as an example
end_date = datetime.now()
start_date = end_date - timedelta(days=7)  # Get data for the last 7 days

stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1m")
prices = stock_data['Close'].values
dates = stock_data.index

# Estimate parameters
dt = 1 / (252 * 390)  # Assuming 252 trading days and 390 minutes per trading day
estimated_params = estimate_params(prices, dt)
theta, mu, sigma = estimated_params
print(f"Estimated parameters: theta={theta:.4f}, mu={mu:.4f}, sigma={sigma:.4f}")

# Generate trading signals
threshold = 0.5 * sigma
signals = trading_signals(prices, mu, threshold)

# Backtest strategy
cumulative_returns = backtest_strategy(prices, signals)

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot stock price
ax1.plot(dates, prices, label='Stock Price')
ax1.set_ylabel('Price')

# Plot mean and thresholds
ax1.axhline(y=mu, color='r', linestyle='--', label='Mean')
ax1.axhline(y=mu + threshold, color='g', linestyle='--', label='Upper Threshold')
ax1.axhline(y=mu - threshold, color='g', linestyle='--', label='Lower Threshold')

ax1.legend()
ax1.set_title(f'{ticker} Stock Price and Trading Signals (1-minute intervals)')

# Plot cumulative returns
ax2.plot(dates[1:], cumulative_returns)
ax2.set_title('Cumulative Returns of Trading Strategy')
ax2.set_xlabel('Date')
ax2.set_ylabel('Cumulative Returns')

plt.tight_layout()
plt.show()

# Calculate strategy performance
total_return = cumulative_returns[-1]
annualized_return = (1 + total_return) ** (252 * 390 / len(cumulative_returns)) - 1
sharpe_ratio = np.sqrt(252 * 390) * np.mean(cumulative_returns) / np.std(cumulative_returns)

print(f"Total Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")