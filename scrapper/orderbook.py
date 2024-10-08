import requests
import pandas as pd

def get_order_book(stock_id):
    # Define the URL for the market depth API
    url = f"https://nepalstock.com.np/api/nots/nepse-data/marketdepth/{stock_id}"

    # Send the GET request to the API
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Debug print the entire response for inspection
        print(f"Response Data for Stock ID {stock_id}:", data)

        # Extract buy and sell market depth
        buy_orders = data['marketDepth']['buyMarketDepthList']
        sell_orders = data['marketDepth']['sellMarketDepthList']

        # Convert buy orders to DataFrame
        buy_df = pd.DataFrame(buy_orders)
        buy_df['orderType'] = 'Buy'  # Add an order type column

        # Convert sell orders to DataFrame
        sell_df = pd.DataFrame(sell_orders)
        sell_df['orderType'] = 'Sell'  # Add an order type column

        # Concatenate buy and sell DataFrames
        order_book_df = pd.concat([buy_df, sell_df], ignore_index=True)

        # Print the combined order book DataFrame
        print(f"Order Book Data for Stock ID {stock_id}:")
        print(order_book_df[['orderBookOrderPrice', 'quantity', 'orderCount', 'orderType']])

        # Save the combined DataFrame to CSV
        #order_book_df.to_csv(f'order_book_stock_{stock_id}.csv', index=False)
        #print(f"Order book data saved to order_book_stock_{stock_id}.csv")
    else:
        print(f"Failed to retrieve data for Stock ID {stock_id}. Status code: {response.status_code}")
        print(response.text)  # Print the response text for debugging

# Example usage: Get order book for a specific stock ID
stock_id = 131  # Replace with the desired stock ID
get_order_book(stock_id)
