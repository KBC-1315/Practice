from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options
import time

def get_google_image_url(query):
    # Chrome WebDriver 경로 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    # Chrome WebDriver 설정
    # service = Service(webdriver_path)

    with webdriver.Chrome(options=options) as driver:
        # Google 이미지 검색 페이지로 이동
        driver.get("https://www.google.com")
        time.sleep(2)  # 페이지 로드를 기다림
        
        # 검색어 입력
        search_box = driver.find_element(By.XPATH, "//input[@name='q']")
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)
        time.sleep(2)  # 검색 결과 로드를 기다림
        
        # 이미지 탭 클릭
        image_tab = driver.find_element(By.XPATH, "//a[contains(@href, 'tbm=isch')]")
        image_tab.click()
        time.sleep(2)  # 이미지 검색 결과 로드를 기다림
        
        # 이미지 URL 가져오기
        first_image = driver.find_element(By.XPATH, "//img[@class='t0fcAb']")
        image_url = first_image.get_attribute('src')
        
    return image_url

# 테스트
query = "Ohtani"
image_url = get_google_image_url(query)
print(image_url)
