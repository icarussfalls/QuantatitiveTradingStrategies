import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

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
        self.spread = 0.005  # 0.5% spread
        self.transaction_cost = 10  # Fixed transaction cost
        
        # Performance tracking
        self.trades = []
        self.profit_loss = 0
    
    def process_trade_data(self, data: pd.DataFrame):
        """
        Process historical trade data to extract market insights
        
        :param data: DataFrame containing trade data
        :return: Tuple of (mid_price, bid_price, ask_price)
        """
        # Filter data for this symbol
        symbol_data = data[data['Symbol'] == self.symbol]
        
        # Calculate mid price
        mid_price = symbol_data['Rate'].mean()
        
        # Calculate bid and ask prices with spread
        bid_price = mid_price * (1 - self.spread/2)
        ask_price = mid_price * (1 + self.spread/2)
        
        return mid_price, bid_price, ask_price
    
    def make_market(self, data: pd.DataFrame) -> List[Dict]:
        """
        Simulate market making strategy
        
        :param data: DataFrame containing trade data
        :return: List of market making trades
        """
        mid_price, bid_price, ask_price = self.process_trade_data(data)
        
        # Market making decisions
        market_making_trades = []
        
        # Bid side trade (buying)
        if self.inventory < self.max_inventory:
            buy_quantity = min(100, self.max_inventory - self.inventory)
            buy_trade = {
                'type': 'BID',
                'price': bid_price,
                'quantity': buy_quantity,
                'total_cost': bid_price * buy_quantity + self.transaction_cost
            }
            
            # Check if we can afford the trade
            if buy_trade['total_cost'] <= self.cash:
                self.cash -= buy_trade['total_cost']
                self.inventory += buy_quantity
                market_making_trades.append(buy_trade)
        
        # Ask side trade (selling)
        if self.inventory > 0:
            sell_quantity = min(100, self.inventory)
            sell_trade = {
                'type': 'ASK',
                'price': ask_price,
                'quantity': sell_quantity,
                'total_revenue': ask_price * sell_quantity - self.transaction_cost
            }
            
            self.cash += sell_trade['total_revenue']
            self.inventory -= sell_quantity
            market_making_trades.append(sell_trade)
        
        # Update profit/loss
        self.profit_loss += sum(trade.get('total_revenue', 0) - trade.get('total_cost', 0) for trade in market_making_trades)
        
        return market_making_trades
    
    def generate_report(self) -> Dict:
        """
        Generate performance report for the market making strategy
        
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

# Initialize and run market making strategy
market_maker = MarketMaker('HRL')
trades = market_maker.make_market(data)

# Print market making trades
print("Market Making Trades:")
for trade in trades:
    print(f"{trade['type']}: {trade['quantity']} shares at {trade['price']}")

# Generate and print performance report
report = market_maker.generate_report()
print("\nPerformance Report:")
for key, value in report.items():
    print(f"{key}: {value}")