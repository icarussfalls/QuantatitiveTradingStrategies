#!/usr/bin/env python
# coding: utf-8

from selenium import webdriver
import os
from webdriver_manager.chrome import ChromeDriverManager
import time
from fake_useragent import UserAgent
from selenium.webdriver.support.ui import Select
import numpy as np
import pandas as pd
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
import warnings
from multiprocessing import Pool
warnings.filterwarnings("ignore")

# Chrome options
options = webdriver.ChromeOptions()

class scrapper():
    def __init__(self, url):
        ua = UserAgent()
        options.add_argument('--no-sandbox')
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f'user-agent={ua.random}')  # add random user agent
        service = Service("/Users/icarus/Downloads/chromedriver-mac-arm64 2/chromedriver")  # replace with your actual path
        self.driver = webdriver.Chrome(service=service, options=options)
        self.url = url
        self.driver.get(url)
        time.sleep(2)
        # Navigate to the history tab
        k = WebDriverWait(self.driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab"]'))).click()
        #self.driver.find_element("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_lnkHistoryTab"]').click()
        #self.driver.implicitly_wait(2)

    def df(self):
        # Extract data from the page
        time.sleep(2)
        Date = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[2]')
        LTP = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[3]')
        Change = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[4]')
        High = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[5]')
        Low = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[6]')
        Open = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[7]')
        Quantity = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[8]')
        Turnover = self.driver.find_elements("xpath", '//*[@id="ctl00_ContentPlaceHolder1_CompanyDetail1_divDataPrice"]/div[2]/table/tbody/tr/td[9]')
        
        # Initialize DataFrame
        data = pd.DataFrame(columns=['Date', 'LTP', 'Change', 'High', 'Low', 'Open', 'Quantity', 'Turnover'])
        
        # Collect the data in a DataFrame
        for i in range(len(Date)):
            data.loc[len(data)] = {
                'Date': Date[i].text,
                'LTP': LTP[i].text,
                'Change': Change[i].text,
                'High': High[i].text,
                'Low': Low[i].text,
                'Open': Open[i].text,
                'Quantity': Quantity[i].text,
                'Turnover': Turnover[i].text
            }
        # if data is sorted according to Date using coerce, will cause incorrect data
        return data

    def datas(self):
        print("Started")
        data = []
        data.append(self.df())

        while True:
            try:
                # Wait until the "Next Page" button is clickable
                k = WebDriverWait(self.driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, '//*[@title = "Next Page"]'))
                )
                
                # Perform the click action once the button is visible
                actions = ActionChains(self.driver)
                actions.move_to_element(k).click().perform()

                # Scrape the new data after moving to the next page
                scrap = self.df()
                data.append(scrap)

            except:
                # If the "Next Page" button is not found, you've reached the last page
                print("No more pages. Finished.")
                break

        # Concatenate all the scraped data into a single DataFrame
        data = pd.concat(data, axis=0)
        
        return data


# Define the stock symbols
stock = ['NABIL']

def save_datas(i):
    x = 'https://merolagani.com/CompanyDetail.aspx?symbol='
    x += str(i)
    y = scrapper(x).datas()
    filename = f"{i}.csv"
    path = os.path.join(os.getcwd(), filename)
    y.to_csv(path, index=None, header=True)
    print(f"Data saved for {i} in {path}")


if __name__ == '__main__':  
        with Pool() as pool:
            pool.map(save_datas, stock)
# call the same function with different data sequentially

