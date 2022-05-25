from selenium import webdriver
from bs4 import BeautifulSoup
import lxml
import time
import numpy as np
import pandas as pd
import requests
import os
import pyautogui
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


directory = os.path.join(os.getcwd(), 'data') 

try:
    if not os.path.exists(directory):
        os.makedirs(directory)
except OSError:
    print ('Error: Creating directory. ' +  directory)

trainpath = os.path.join(directory, 'train') 

try:
    if not os.path.exists(trainpath):
        os.makedirs(trainpath)
except OSError:
    print ('Error: Creating directory. ' +  trainpath)

browser = webdriver.Chrome('chromedriver.exe')
browser.implicitly_wait(3)
year = 2022
month = 5
day = 14
hour = np.arange(0,24)
min = np.arange(0,60, 5)

for h in hour:
    if h < 10:
        h = "0" + str(h)
    for m in min:
        if m < 10:
            m = "0" + str(m)

        browser.get('https://www.weather.go.kr/weather/images/rader_integrate.jsp?autoStart=false&zoomLevel=0&zoomX=0000000&zoomY=0000000&data=SFC-HSR&tm={}.{}.{}.{}%3A{}&timeTerm=10&x=0&y=0&itv=0.5'.format(year, month, day, h, m))

        time.sleep(2)
        imgUrl = browser.find_element_by_xpath('//*[@id="rdr-player"]/div[2]/ul/li/img').get_attribute("src")
        imgtime = browser.find_element_by_xpath('//*[@id="rdr-player"]/div[2]/ul/li/img').get_attribute("title")
        print(f"이미지 주소: {imgUrl}")

        from urllib.request import urlretrieve

        outfile = "test" + str(year)+ str(month)+ str(day) + str(h) + str(m) + ".png"
        outfile = os.path.join(trainpath, outfile)
        urlretrieve(imgUrl, outfile)
        

browser.close()




