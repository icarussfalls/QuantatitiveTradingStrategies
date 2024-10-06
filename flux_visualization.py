import numpy as np

# Constants
price_levels = np.arange(100, 200, 1)  # Prices from 100 to 200
time_steps = 100  # Total simulation time steps
order_book_volume = np.zeros_like(price_levels)  # Initial order volume at each price level

# Populate the order book with some initial volume
order_book_volume += np.random.randint(5, 15, size=len(price_levels))

# Parameters for market making
initial_cash = 100000  # Starting cash for the market maker
inventory = 0  # Inventory of the asset (number of units held)
buy_spread = 1  # Distance below mid-price to place buy orders
sell_spread = 1  # Distance above mid-price to place sell orders
order_size = 10  # Number of units in each buy/sell order

# Function to simulate order influx
def influx(volume, price_levels, rate):
    price_idx = np.random.choice(len(price_levels), size=rate)
    for idx in price_idx:
        volume[idx] += np.random.randint(1, 10)
    return volume

# Function to simulate order outflux
def outflux(volume, rate):
    non_empty_idx = np.where(volume > 0)[0]
    if len(non_empty_idx) > 0:
        price_idx = np.random.choice(non_empty_idx, size=min(rate, len(non_empty_idx)))
        for idx in price_idx:
            volume[idx] = max(0, volume[idx] - np.random.randint(1, 10))
    return volume

# Function to find the best bid and ask
def best_bid_ask(volume, price_levels):
    non_zero_volumes = np.where(volume > 0)[0]
    if len(non_zero_volumes) == 0:
        return None, None  # No orders in the book
    best_bid = price_levels[non_zero_volumes[0]]  # Highest buy price (bid)
    best_ask = price_levels[non_zero_volumes[-1]]  # Lowest sell price (ask)
    return best_bid, best_ask

# Market making function
def market_maker_strategy(volume, price_levels, cash, inventory):
    # Find best bid and ask
    best_bid, best_ask = best_bid_ask(volume, price_levels)
    
    if best_bid is None or best_ask is None:
        print("No active bids or asks in the order book")
        return cash, inventory

    # Calculate mid price
    mid_price = (best_bid + best_ask) / 2

    # Place buy and sell orders
    buy_price = mid_price - buy_spread  # Buy spread below mid-price
    sell_price = mid_price + sell_spread  # Sell spread above mid-price

    # Round buy and sell prices to the nearest valid price level
    buy_price_idx = np.searchsorted(price_levels, buy_price, side="left")
    sell_price_idx = np.searchsorted(price_levels, sell_price, side="right") - 1

    # Ensure we don't exceed price boundaries
    buy_price_idx = np.clip(buy_price_idx, 0, len(price_levels) - 1)
    sell_price_idx = np.clip(sell_price_idx, 0, len(price_levels) - 1)

    buy_price = price_levels[buy_price_idx]
    sell_price = price_levels[sell_price_idx]

    print(f"Market maker places buy at {buy_price} and sell at {sell_price}")

    # Execute the buy order if there's volume at or below the buy price
    if order_book_volume[buy_price_idx] > 0 and cash >= buy_price * order_size:
        volume[buy_price_idx] -= order_size  # Reduce order book volume
        cash -= buy_price * order_size  # Spend cash
        inventory += order_size  # Increase inventory
        print(f"Executed buy order at {buy_price}")

    # Execute the sell order if there's volume at or above the sell price
    if inventory >= order_size and order_book_volume[sell_price_idx] > 0:
        volume[sell_price_idx] -= order_size  # Reduce order book volume
        cash += sell_price * order_size  # Gain cash
        inventory -= order_size  # Decrease inventory
        print(f"Executed sell order at {sell_price}")

    return cash, inventory

# Simulate the market-making strategy over time
cash = initial_cash
for t in range(time_steps):
    print(f"Time step {t+1}")

    # Simulate increased order influx and outflux to create more dynamic market conditions
    order_book_volume = influx(order_book_volume, price_levels, rate=10)
    order_book_volume = outflux(order_book_volume, rate=5)

    # Apply market-making strategy with adjusted parameters
    cash, inventory = market_maker_strategy(order_book_volume, price_levels, cash, inventory)

# Final cash and inventory
print(f"Final cash: {cash}, Final inventory: {inventory}")


