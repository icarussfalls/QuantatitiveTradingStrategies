import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def generate_macd_signals(data, short_window=12, long_window=26, signal_window=9):
    signals = pd.DataFrame(index=data.index)
    signals['MACD'], signals['Signal_Line'], signals['MACD_Histogram'] = calculate_macd(data)

    # Buy when MACD crosses above Signal Line
    signals['Buy_Signal'] = np.where((signals['MACD'] > signals['Signal_Line']) & (signals['MACD'].shift() <= signals['Signal_Line'].shift()), 1, 0)
    # Sell when MACD crosses below Signal Line
    signals['Sell_Signal'] = np.where((signals['MACD'] < signals['Signal_Line']), 1, 0)
    return signals

def generate_rsi_signals(data, rsi_oversold=30, rsi_overbought=70):
    signals = pd.DataFrame(index=data.index)
    signals['RSI'] = calculate_rsi(data)

    # Buy when RSI crosses above oversold threshold
    signals['Buy_Signal'] = np.where((signals['RSI'] > rsi_oversold) & (signals['RSI'].shift() <= rsi_oversold), 1, 0)
    # Sell when RSI crosses below overbought threshold
    signals['Sell_Signal'] = np.where((signals['RSI'] < rsi_overbought), 1, 0)
    return signals

def backtest_strategy(data, signals, strategy_name, min_holding_period=3, commission=0.001):
    initial_balance = 10000
    balance = initial_balance
    position = 0
    buy_price = 0
    holding_period = 0
    portfolio_value = []

    total_shares_bought = 0
    total_shares_sold = 0
    total_profit = 0
    peak_balance = initial_balance
    max_drawdown = 0

    for index, row in signals.iterrows():
        current_price = data.loc[index, 'Close']
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]

        if row['Buy_Signal'] == 1 and position == 0:
            shares_to_buy = (balance // (current_price * (1 + commission))) // 10 * 10
            if shares_to_buy > 0:
                position = shares_to_buy
                buy_price = current_price
                balance -= position * current_price * (1 + commission)
                holding_period = 0
                total_shares_bought += shares_to_buy

        elif row['Sell_Signal'] == 1 and position > 0 and holding_period >= min_holding_period:
            sale_value = position * current_price * (1 - commission)
            profit = sale_value - (position * buy_price * (1 + commission))
            balance += sale_value
            total_shares_sold += position
            total_profit += profit
            position = 0

        portfolio_value.append(balance + position * current_price)
        peak_balance = max(peak_balance, balance + position * current_price)
        current_drawdown = (peak_balance - (balance + position * current_price)) / peak_balance
        max_drawdown = max(max_drawdown, current_drawdown)

        if position > 0:
            holding_period += 1

    if position > 0:
        final_price = data.iloc[-1]['Close']
        balance += position * final_price * (1 - commission)
        portfolio_value[-1] = balance

    final_portfolio_value = balance
    portfolio_value_series = pd.Series(portfolio_value)
    daily_returns = portfolio_value_series.pct_change().dropna()
    total_return = (final_portfolio_value - initial_balance) / initial_balance
    risk_free_rate = 0.02

    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0

    return {
        "Strategy": strategy_name,
        "Initial Balance": initial_balance,
        "Final Balance": final_portfolio_value,
        "Total Shares Bought": total_shares_bought,
        "Total Shares Sold": total_shares_sold,
        "Total Profit": total_profit,
        "Max Drawdown (%)": max_drawdown * 100,
        "Sharpe Ratio": sharpe_ratio
    }, portfolio_value

# Load your data
data = pd.read_csv('/Users/icarus/Desktop/QuantatitiveTradingStrategies/NFS.csv')
data = data.sort_values(by='Date', ascending=True)
data = data.set_index('Date')
data['Close'] = pd.to_numeric(data['LTP'], errors='coerce')
data.dropna(inplace=True)

# Generate signals for MACD and RSI
signals_macd = generate_macd_signals(data)
signals_rsi = generate_rsi_signals(data)

# Backtest each strategy
results_macd, portfolio_macd = backtest_strategy(data, signals_macd, "MACD Strategy")
results_rsi, portfolio_rsi = backtest_strategy(data, signals_rsi, "RSI Strategy")

# Combine both strategies
signals_combined = pd.DataFrame(index=data.index)
signals_combined['Buy_Signal'] = np.where((signals_rsi['Buy_Signal'] == 1) & (signals_macd['Buy_Signal'] == 1), 1, 0)
signals_combined['Sell_Signal'] = np.where((signals_rsi['Sell_Signal'] == 1) & (signals_macd['Sell_Signal'] == 1), 1, 0)

# Backtest the combined strategy
results_combined, portfolio_combined = backtest_strategy(data, signals_combined, "Combined MACD + RSI Strategy")

# Create a DataFrame to summarize results
results_df = pd.DataFrame([results_macd, results_rsi, results_combined])
print(results_df)

# Plot All Strategies in a Single Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, portfolio_macd, label="MACD Strategy Portfolio Value", color='b')
plt.plot(data.index, portfolio_rsi, label="RSI Strategy Portfolio Value", color='g')
plt.plot(data.index, portfolio_combined, label="Combined MACD + RSI Strategy Portfolio Value", color='r')

plt.title('Portfolio Value of MACD, RSI, and Combined Strategies')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()