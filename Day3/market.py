import pandas as pd
import numpy as np
from typing import List, Dict


class MarketMaker:
    def __init__(self, symbol: str):
        """
        Initialize the Market Maker for a specific stock symbol
        
        :param symbol: Stock symbol to trade
        """
        self.symbol = symbol
        
        # Trading parameters
        self.inventory = 0  # Current stock inventory
        self.cash = 100000  # Starting cash
        self.max_inventory = 1000  # Maximum inventory limit
        
        # Market making strategy parameters
        self.spread = 0.02  # 2% spread (Increased spread)
        self.transaction_cost = 10  # Fixed transaction cost
        
        # Performance tracking
        self.trades: List[Dict] = []
        self.profit_loss = 0

    def calculate_prices(self, current_price: float) -> Dict[str, float]:
        """
        Calculate mid-price, bid, and ask prices based on current market price.
        
        :param current_price: Latest market price
        :return: Dictionary containing mid_price, bid_price, and ask_price
        """
        mid_price = current_price  # In a live scenario, the current price is used as the mid price.
        bid_price = mid_price * (1 - self.spread / 2)
        ask_price = mid_price * (1 + self.spread / 2)
        
        return {'mid_price': mid_price, 'bid_price': bid_price, 'ask_price': ask_price}

    def market_making(self, data: pd.DataFrame) -> List[Dict]:
        """
        Execute the market-making strategy for the data

        :param data: DataFrame containing trade data
        :return: List of trades made
        """
        market_making_trades = []
        
        # Iterate over the rows in chunks of 10
        for i in range(0, len(data) - 10, 10):
            # Get the 10 rows for market-making
            floor_data = data.iloc[i:i + 10]
            mid_price = floor_data['Rate'].mean()  # Calculate mid price as the average of the 10 prices

            # Calculate bid and ask based on the mid price
            bid_price = mid_price * (1 - self.spread / 2)
            ask_price = mid_price * (1 + self.spread / 2)

            print(f"Processed block {i//10 + 1}: Mid Price: {mid_price}, Bid Price: {bid_price}, Ask Price: {ask_price}")
            
            # Monitor subsequent rows to see if any price hits the bid or ask
            canceled_bid_ask = False  # Flag to track if the bid/ask is canceled
            
            for j in range(i + 10, i + 20):  # Monitor the next 10 rows
                if j >= len(data):
                    break

                current_price = data.loc[j, 'Rate']
                
                print(f"Current Price: {current_price}, Checking Bid: {current_price <= bid_price}, Checking Ask: {current_price >= ask_price}")

                if current_price <= bid_price and self.inventory < self.max_inventory:
                    # Calculate the maximum number of shares we can buy with available cash
                    max_shares_to_buy = (self.cash - self.transaction_cost) // bid_price  # Floor the amount of shares based on cash available
                    
                    # Limit the number of shares to the smaller of the max we can buy and the inventory limit
                    buy_quantity = min(max_shares_to_buy, 100, self.max_inventory - self.inventory)

                    buy_trade = {
                        'type': 'BID',
                        'price': bid_price,
                        'quantity': buy_quantity,
                        'total_cost': bid_price * buy_quantity + self.transaction_cost
                    }

                    # Ensure there is enough cash to buy
                    if buy_trade['total_cost'] <= self.cash:
                        self.cash -= buy_trade['total_cost']
                        self.inventory += buy_quantity
                        market_making_trades.append(buy_trade)
                        print(f"Executing Buy: {buy_quantity} at {bid_price}")
                    else:
                        print(f"Not enough cash to buy: {buy_trade['total_cost']} > {self.cash}")

                elif current_price >= ask_price and self.inventory > 0:
                    # Execute a sell at the ask price
                    sell_quantity = min(100, self.inventory)
                    sell_trade = {
                        'type': 'ASK',
                        'price': ask_price,
                        'quantity': sell_quantity,
                        'total_revenue': ask_price * sell_quantity - self.transaction_cost
                    }

                    # Add revenue from sell trade to cash
                    self.cash += sell_trade['total_revenue']
                    self.inventory -= sell_quantity
                    market_making_trades.append(sell_trade)
                    print(f"Executing Sell: {sell_quantity} at {ask_price}")
            
            # If the bid and ask price were not hit in the next 10 rows, cancel the orders
            if not canceled_bid_ask:
                print(f"Canceling Bid and Ask orders for {self.symbol} at {bid_price} and {ask_price}")
                self.cash += self.transaction_cost  # Refund the transaction cost
                canceled_bid_ask = True

        # Update profit/loss
        self.profit_loss += sum(trade.get('total_revenue', 0) - trade.get('total_cost', 0) for trade in market_making_trades)

        return market_making_trades

    def generate_report(self) -> Dict:
        """
        Generate performance report for the market-making strategy
        
        :return: Dictionary with performance metrics
        """
        return {
            'symbol': self.symbol,
            'final_cash': self.cash,
            'final_inventory': self.inventory,
            'total_profit_loss': self.profit_loss
        }


# Load the data
data = pd.read_csv('Day3/HRL_combined.csv', thousands=',')

# Initialize and run market-making strategy
market_maker = MarketMaker('HRL')
trades = market_maker.market_making(data)

# Print market-making trades
print("\nMarket Making Trades:")
for trade in trades:
    print(f"{trade['type']}: {trade['quantity']} shares at {trade['price']}")

# Generate and print performance report
report = market_maker.generate_report()
print("\nPerformance Report:")
for key, value in report.items():
    print(f"{key}: {value}")
