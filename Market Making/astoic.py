import numpy as np
import matplotlib.pyplot as plt

# Parameters
S = 100  # Mid-price of the asset
sigma = 2  # Asset price volatility
k = 1.5  # Market order arrival rate
gamma = 0.1  # Risk aversion parameter
q = 0  # Initial inventory level
time_horizon = 1  # 1 day
dt = 1 / 252  # Time step (daily for simplicity)
lambda_buy = 0.5  # Poisson arrival rate for buy orders
lambda_sell = 0.5  # Poisson arrival rate for sell orders
mu = 0.01  # Drift in price (e.g., expected return)
depth_factor = 0.1  # Order book depth impact
inventory_penalty_factor = 0.01  # Penalty for large inventory levels

# Spread adjustment
def calculate_spread(k, lambda_=1.0):
    return 0.5 * np.log(1 + k / lambda_)

# Optimal bid and ask prices with inventory penalty and order book depth
def avellaneda_stoikov_pricing(S, sigma, q, k, gamma, dt, depth_factor, inventory_penalty_factor):
    delta = calculate_spread(k)
    inventory_penalty = inventory_penalty_factor * q**2  # Quadratic penalty
    order_book_depth_adjustment = depth_factor * q
    bid_price = S - delta - (gamma * sigma**2 * q) / (2 * k) - inventory_penalty - order_book_depth_adjustment
    ask_price = S + delta - (gamma * sigma**2 * q) / (2 * k) + inventory_penalty - order_book_depth_adjustment
    return bid_price, ask_price

# Simulate Poisson arrivals
def simulate_poisson_arrivals(rate, time_horizon, dt):
    time_steps = int(time_horizon / dt)
    return np.random.poisson(rate * dt, time_steps)

# Run simulation
time_steps = int(time_horizon / dt)
inventory = [q]
prices = [S]
bid_prices = []
ask_prices = []

# Simulate the market-making process
for t in range(time_steps):
    # Compute bid and ask prices
    bid_price, ask_price = avellaneda_stoikov_pricing(
        S, sigma, inventory[-1], k, gamma, dt, depth_factor, inventory_penalty_factor
    )
    bid_prices.append(bid_price)
    ask_prices.append(ask_price)
    
    # Simulate order arrivals (Poisson process)
    buy_orders = simulate_poisson_arrivals(lambda_buy, dt, dt)
    sell_orders = simulate_poisson_arrivals(lambda_sell, dt, dt)
    
    # Process buy orders (if price <= bid price, increase inventory)
    if buy_orders and np.random.uniform(0, 1) <= 0.5:  # Random execution
        inventory.append(inventory[-1] + 1)  # Fill buy order
    elif sell_orders and np.random.uniform(0, 1) <= 0.5:  # Process sell orders
        inventory.append(inventory[-1] - 1)  # Fill sell order
    else:
        inventory.append(inventory[-1])  # No change in inventory

    # Simulate mid-price changes using GBM
    dS = mu * S * dt + sigma * S * np.sqrt(dt) * np.random.normal()
    S += dS
    prices.append(S)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(prices, label="Mid-Price")
plt.plot(bid_prices, label="Bid Price", linestyle="--")
plt.plot(ask_prices, label="Ask Price", linestyle="--")
plt.xlabel("Time Steps")
plt.ylabel("Price")
plt.title("Enhanced Avellaneda-Stoikov with Realistic Features")
plt.legend()
plt.grid()
plt.show()

# Plot inventory over time
plt.figure(figsize=(12, 4))
plt.plot(inventory, label="Inventory Level")
plt.xlabel("Time Steps")
plt.ylabel("Inventory")
plt.title("Inventory Management with Penalties")
plt.legend()
plt.grid()
plt.show()
