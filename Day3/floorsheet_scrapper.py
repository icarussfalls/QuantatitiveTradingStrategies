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

class FloorsheetScrapper:
    def __init__(self, stock, url, date):
        ua = UserAgent()
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f'user-agent={ua.random}')
        service = Service("/Users/icarus/Downloads/chromedriver-mac-arm64 2/chromedriver")  # Adjust the path as needed
        self.driver = webdriver.Chrome(service=service, options=options)
        self.url = url
        self.driver.get(url)
        self.stock = stock
        
        # Input stock symbol
        self.driver.find_element(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_ASCompanyFilter_txtAutoSuggest"]').send_keys(self.stock)
        
        # Input date in mm/dd/yyyy format
        date_input_xpath = '//*[@id="ctl00_ContentPlaceHolder1_txtFloorsheetDateFilter"]'
        self.driver.find_element(By.XPATH, date_input_xpath).send_keys(date)
        
        # Click the search button
        self.driver.find_element(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_lbtnSearchFloorsheet"]').click()

    def attempt_scraping(self):
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

def collect_data(symbol, url, start_date, end_date):
    current_date = start_date
    all_data = []
    
    # Ensure the Day3 directory exists
    day3_folder = os.path.join(os.getcwd(), "Day3")
    os.makedirs(day3_folder, exist_ok=True)

    while current_date <= end_date:
        date_str = current_date.strftime('%m/%d/%Y')
        print(f"Checking data for {date_str}")

        try:
            scraper = FloorsheetScrapper(symbol, url, date_str)
            data = scraper.scraped_data()

            # Check if data is found
            if not data.empty:
                all_data.append(data)
                filename = f"{symbol}_{date_str.replace('/', '-')}.csv"
                path = os.path.join(day3_folder, filename)  # Save inside Day3 folder
                data.to_csv(path, index=False)
                print(f"Data saved for {symbol} on {date_str} in {path}")

        except Exception as e:
            print(f"Error occurred while scraping data for {date_str}: {e}")

        # Move to the next day
        current_date += timedelta(days=1)

    # Concatenate all collected data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, axis=0)
        combined_filename = f"{symbol}_combined.csv"
        combined_path = os.path.join(day3_folder, combined_filename)  # Save combined data in Day3 folder
        combined_data.to_csv(combined_path, index=False)
        print(f"All collected data saved in {combined_path}")

# Usage
url = 'https://merolagani.com/Floorsheet.aspx'
symbol = 'NABIL'

# Define the date range
end_date = datetime.today()
start_date = end_date - timedelta(days=7)  # One week back

collect_data(symbol, url, start_date, end_date)
