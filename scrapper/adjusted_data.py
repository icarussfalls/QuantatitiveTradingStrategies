import requests
import json
import pandas as pd
import os

stocks_banking = ['NABIL', 'KBL', 'MBL', 'SANIMA', 'NICA']
stocks_finance = ['CFCL', 'GFCL', 'MFIL', 'GUFL', 'NFL']
stocks_microfinance = ['CBBL', 'DDBL', 'SKBBL', 'SMFDB', 'SWBBL', 'SMB', 'FOWAD', 'KLBSL']
stocks_life = ['ALICL', 'LICN', 'NLIC', 'CLI', 'ILI', 'SJLIC']
stocks_non_life = ['NICL', 'NIL', 'NLG', 'SICL', 'PRIN', 'HEI']
stocks_others = ['NRIC', 'NTC', 'SHIVM', 'HRL']
stocks_hydro = ['NYADI', 'RADHI', 'NHPC', 'KPCL', 'HDHPC', 'DHPL', 'API', 'AKPL', 'UNHPL']

stock_symbols = stocks_banking + stocks_finance + stocks_microfinance + stocks_life + stocks_non_life + stocks_hydro + stocks_others

# Base URL for the request
url = "https://nepsealpha.com/trading/1/history"

# Set up headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://nepsealpha.com/trading/chart?symbol=',
    'X-Requested-With': 'XMLHttpRequest',
    'DNT': '1'
}

# Loop through each stock symbol and retrieve data
for symbol in stock_symbols:
    # Define the payload for each stock symbol
    payload = {
        'fsk': '1728308583040',
        'symbol': symbol,
        'resolution': '1D',
        'pass': 'ok'
    }
    
    try:
        # Send a GET request
        response = requests.get(url, params=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = json.loads(response.text)

            # Create a DataFrame from the data
            df = pd.DataFrame(data)

            # Rename the columns
            df.columns = ['Status', 'Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']

            # Convert the OHLCV columns to numeric
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['High'] = pd.to_numeric(df['High'], errors='coerce')
            df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')


            # Drop the 'Status' column
            df = df.drop(columns=['Status'])

            # Convert the timestamp to datetime
            df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')

            # Drop the original Timestamp column
            df = df.drop(columns=['Timestamp'])


            df_sorted = df.sort_values(by='Date', ascending=True)

            filename = f"{symbol}.csv"
            # Save DataFrame to CSV
            path = os.path.join(os.getcwd(), 'datas/' + filename)
            df_sorted.to_csv(path, index=False)
            print(f"Data saved for {symbol} in {path}")
            print(df_sorted)
        else:
            print(f"Failed to retrieve data for {symbol}: {response.status_code}")
    except Exception as e:
        print(f"Error processing {symbol}: {e}")
