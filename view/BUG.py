from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time


class Bug(object):
    def __init__(self):
        self.number = "MPT-0966"
        self.path = "chromedriver.exe"

        self.options = Options()
        self.prefs = {"download.default_directory": self.path}
        self.options.add_experimental_option("prefs", self.prefs)
        self.browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=self.options)
    
    def start_B(self):
        self.browser.get("https://mobile.epa.gov.tw/motor/query/query_check.aspx")
        time.sleep(1)
        self.search_1 = self.browser.find_element(By.NAME, "ctl00$MainContent$txtCarNo")
        self.search_1.click()
        self.search_1.send_keys("MPT-0966")
        time.sleep(1)

        self.search_2 = self.browser.find_element(By.NAME, "ctl00$MainContent$btnQuery")
        self.search_2.click()
        time.sleep(1)
        self.search_3 = self.browser.find_element(By.XPATH, "/html/body/form/div[4]/div[3]/div/div[3]/div[1]/table/tbody/tr[2]")

        self.search_3 = self.search_3.text
        # search_3 = search_3["檢驗別"]
        self.search_3 = self.search_3.split(" ")
        # print(self.search_3)
        # self.vehicle = {"車牌號碼":self.search_3[0],
        #                 # "是否檢測":self.search_3[1],
        #                 # "檢測結果":self.search_3[2],
        #                 "HC":self.search_3[2],
        #                 "CO":self.search_3[3],
        #                 "CC":self.search_3[4],
        #                 "檢測結果":self.search_3[6],
        #                 # "型程別":self.search_3[6],
        #                 "檢測日期":self.search_3[7]}
        self.vehicle = [self.search_3[0], self.search_3[2], self.search_3[3],
                        self.search_3[4], self.search_3[6], self.search_3[7]]
        print(self.vehicle)
        # time.sleep(1)
        # # self.browser.close()
        # time.sleep(1)
        # self.open_web()

    # def second(self):
    #     self.browser.get("https://polcar.epa.gov.tw/case/Step0.aspx")
    #     time.sleep(1)

    def open_web(self):
        self.browser.get("https://polcar.epa.gov.tw/case/Step0.aspx")

        time.sleep(1)
        search_1 = self.browser.find_element(By.XPATH, "/html/body/form/main/div/div/table/tbody/tr[1]/td/label")
        search_1.click()

        time.sleep(1)
        search_2 = self.browser.find_element(By.XPATH, "/html/body/form/main/div/div/p[28]/input")
        search_2.click()

        # time.sleep(3)
        # search_3 = self.browser.find_element(By.XPATH, "/html/body/form/main/div/div/p[28]/input")
        # search_3.click()

        time.sleep(2)
    
        self.search_4 = self.browser.find_element(By.XPATH, "/html/body/form/main/div/div/article/div/table/tbody/tr/td/input[1]")
        self.search_4.click()
        self.search_4.send_keys(self.number[:3])
        self.search_5 = self.browser.find_element(By.XPATH, "/html/body/form/main/div/div/article/div/table/tbody/tr/td/input[2]")
        self.search_5.click()
        self.search_5.send_keys(self.number[4:])
        self.search_6 = self.browser.find_element(By.XPATH, "/html/body/form/main/div/div/article/div/table/tbody/tr/td/input[3]")
        self.search_6.click()
        time.sleep(10)
        # self.browser.close()

# test = Bug()
# test.start()