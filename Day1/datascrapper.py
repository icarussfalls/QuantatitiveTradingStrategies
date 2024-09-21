#!/usr/bin/env python
# coding: utf-8

from selenium import webdriver
import os
import time
from fake_useragent import UserAgent
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from multiprocessing import Process
import warnings

warnings.filterwarnings("ignore")

# Chrome options
options = webdriver.ChromeOptions()

class Scrapper():
    def __init__(self, url):
        ua = UserAgent()
        options.add_argument('--no-sandbox')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f'user-agent={ua.random}')  # Add random user agent
        service = Service("/Users/icarus/Downloads/chromedriver-mac-arm64 2/chromedriver")  # Your specific path
        self.driver = webdriver.Chrome(service=service, options=options)
        self.url = url
        self.driver.get(url)
        # Navigate to the history tab
        self.driver.find_element("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab"]').click()
        self.driver.implicitly_wait(2)

    def df(self):
        attempt = 1  # Initialize attempt counter
        while True:  # Infinite loop, will break when successful
            try:
                # Wait for the table to load
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr'))
                )
                
                # Extract all rows from the table
                rows = self.driver.find_elements(By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr')
                
                # Initialize list to collect data
                data = []
                
                # Iterate over rows and extract the required data
                for row in rows:
                    cols = row.find_elements(By.TAG_NAME, "td")
                    if len(cols) >= 9:  # Ensure there are enough columns
                        data.append({
                            'Date': cols[1].text,
                            'LTP': cols[2].text,
                            'Change': cols[3].text,
                            'High': cols[4].text,
                            'Low': cols[5].text,
                            'Open': cols[6].text,
                            'Quantity': cols[7].text,
                            'Turnover': cols[8].text
                        })
                
                # Create DataFrame from the collected data
                df = pd.DataFrame(data)

                # Convert the 'Date' column to a datetime object for proper sorting
                df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')  # Adjust format as necessary

                # Sort the DataFrame by 'Date' in ascending order (oldest to newest)
                df = df.sort_values(by='Date', ascending=True)

                # Break out of the retry loop if no exception occurred
                break
            
            except StaleElementReferenceException as e:
                print(f"Stale element encountered, retrying... (attempt {attempt})")
                attempt += 1  # Increment the attempt counter
                time.sleep(2)  # Wait before retrying

        # Return the DataFrame after successful extraction
        return df




    def datas(self):
        print("Started scraping for:", self.url)
        data = []
        data.append(self.df())

        while True:
            try:
                # Wait until the "Next Page" button is clickable
                k = WebDriverWait(self.driver, 20).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@title = "Next Page"]'))
                )
                
                # Perform the click action once the button is visible
                actions = ActionChains(self.driver)
                actions.move_to_element(k).click().perform()

                # Scrape the new data after moving to the next page
                scrap = self.df()
                data.append(scrap)

            except Exception as e:
                print("No more pages. Finished. Error:", e)
                break

        # Concatenate all the scraped data into a single DataFrame
        data = pd.concat(data, axis=0)
        
        return data

def save_datas(symbol):
    url = f'https://merolagani.com/CompanyDetail.aspx?symbol={symbol}'
    scrapper = Scrapper(url)
    data = scrapper.datas()
    filename = f"{symbol}.csv"
    path = os.path.join(os.getcwd(), filename)
    data.to_csv(path, index=False)
    print(f"Data saved for {symbol} in {path}")

if __name__ == '__main__':
    stock_symbols = ['KBL', 'EBL']
    processes = []

    for symbol in stock_symbols:
        process = Process(target=save_datas, args=(symbol,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()  # Wait for all processes to finish
