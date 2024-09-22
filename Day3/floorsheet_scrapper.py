from datetime import datetime, timedelta
from selenium import webdriver
import os
import pandas as pd
from fake_useragent import UserAgent
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from concurrent.futures import ProcessPoolExecutor, as_completed

url = 'https://merolagani.com/Floorsheet.aspx'

class FloorsheetScrapper:
    def __init__(self, stock, url):
        ua = UserAgent()
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f'user-agent={ua.random}')
        service = Service("/Users/icarus/Downloads/chromedriver-mac-arm64 2/chromedriver")  # Adjust the path
        self.driver = webdriver.Chrome(service=service, options=options)
        self.url = url
        self.driver.get(url)
        self.stock = stock

        # Input stock symbol
        self.driver.find_element(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_ASCompanyFilter_txtAutoSuggest"]').send_keys(self.stock)
    
    def set_date_and_search(self, date):
        # Input date in mm/dd/yyyy format
        date_input_xpath = '//*[@id="ctl00_ContentPlaceHolder1_txtFloorsheetDateFilter"]'
        date_input = self.driver.find_element(By.XPATH, date_input_xpath)
        date_input.clear()
        date_input.send_keys(date)

        # Click the search button
        self.driver.find_element(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_lbtnSearchFloorsheet"]').click()

    def close_popup(self):
        try:
            # Adjust the XPath according to the actual pop-up structure
            popup_close_button_xpath = '//*[@id="dismiss-button"]/div/span'  # Actual XPath to close Ad
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, popup_close_button_xpath))
            ).click()
            print("Pop-up closed.")
        except NoSuchElementException:
            print("No pop-up to close.")
        except Exception as e:
            print(f"Error while trying to close pop-up: {e}")

    def attempt_scraping(self):
        self.close_popup()  # Close pop-up before scraping
        table_xpath = '//*[@id="ctl00_ContentPlaceHolder1_divData"]/div[4]/table/tbody/'
        rows = self.driver.find_elements(By.XPATH, f"{table_xpath}tr")

        if not rows:
            return pd.DataFrame()  # Return empty DataFrame if no rows

        table = []
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            row_data = [col.text for col in cols]
            table.append(row_data)

        df = pd.DataFrame(table)
        df.rename(columns={1: 'Transact_No', 2: 'Symbol', 3: 'Buyer', 4: 'Seller', 5: 'Quantity', 6: 'Rate', 7: 'Amount'}, inplace=True)
        return df

    def move_to_next_page(self):
        try:
            next_page_xpath = '//*[@title = "Next Page"]'
            WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, next_page_xpath))
            ).click()
            return True
        except (NoSuchElementException, Exception):
            print("No more pages or error while navigating.")
            return False

    def scraped_data(self):
        data = []
        while True:
            scrap = self.attempt_scraping()
            if scrap.empty:
                break
            data.append(scrap)

            if not self.move_to_next_page():
                break
        
        # Concatenate all the scraped data into a single DataFrame
        if data:
            return pd.concat(data, axis=0)
        return pd.DataFrame()  # Return empty DataFrame if no data found

    def close(self):
        self.driver.quit()

def collect_data_for_stock(args):
    symbol, url, start_date, end_date = args
    all_data = []

    try:
        scraper = FloorsheetScrapper(symbol, url)

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%m/%d/%Y')
            print(f"Checking data for {symbol} on {date_str}")

            try:
                scraper.set_date_and_search(date_str)
                data = scraper.scraped_data()

                # Check if data is found
                if not data.empty:
                    all_data.append(data)
                    filename = f"{symbol}_{date_str.replace('/', '-')}.csv"
                    path = os.path.join(os.getcwd(), 'Day3', filename)  # Save inside Day3 folder
                    data.to_csv(path, index=False)
                    print(f"Data saved for {symbol} on {date_str} in {path}")

            except Exception as e:
                print(f"Error occurred while scraping data for {symbol} on {date_str}: {e}")

            # Move to the next day
            current_date += timedelta(days=1)

        scraper.close()

    except Exception as e:
        print(f"Error setting up scraper for {symbol}: {e}")

    return all_data

def collect_data(symbols, url, start_date, end_date):
    # Create the Day3 directory if it doesn't exist
    os.makedirs(os.path.join(os.getcwd(), 'Day3'), exist_ok=True)

    # Prepare arguments for multiprocessing
    args = [(symbol, url, start_date, end_date) for symbol in symbols]

    with ProcessPoolExecutor(max_workers=len(symbols)) as executor:
        futures = {executor.submit(collect_data_for_stock, arg): arg for arg in args}
        all_combined_data = []

        for future in as_completed(futures):
            result = future.result()
            all_combined_data.extend(result)

    # Combine all collected data into individual DataFrames for each stock
    for symbol in symbols:
        symbol_data = [df for df in all_combined_data if not df.empty and df['Symbol'].iloc[0] == symbol]
        if symbol_data:
            combined_data = pd.concat(symbol_data, axis=0)

            # Save combined data for the symbol
            combined_filename = f"{symbol}_combined.csv"
            combined_path = os.path.join(os.getcwd(), 'Day3', combined_filename)
            combined_data.to_csv(combined_path, index=False)
            print(f"All collected data for {symbol} saved in {combined_path}")

# Usage
url = 'https://merolagani.com/Floorsheet.aspx'

if __name__ == '__main__':
    # Define stock symbols and the date range
    symbols = ['NGPL']  # Add more stock symbols as needed

    # Define the date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=7)  # One week back

    collect_data(symbols, url, start_date, end_date)
