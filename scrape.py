from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up the Chrome web driver
s = Service('C:\Program Files\Google\Chrome\ChromeDriver\chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # run Chrome in headless mode (without a visible window)
driver = webdriver.Chrome(service=s, options=options)

# Navigate to the "All Memes" page on Know Your Meme
driver.get('https://knowyourmeme.com/memes/all')

# Wait for the page to load and scroll down to load all the meme entries
wait = WebDriverWait(driver, 30)
wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.entry[class*="entry"]')))

while True:
    prev_height = driver.execute_script("return document.body.scrollHeight")
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    wait.until(lambda driver: driver.execute_script("return document.body.scrollHeight") > prev_height)
    if driver.execute_script("return window.pageYOffset + window.innerHeight >= document.body.scrollHeight"):
        break

# Extract the names of all the memes
meme_names = []
for meme in driver.find_elements(By.CSS_SELECTOR, '.entry[class*="entry"]'):
    meme_names.append(meme.text)

# Print the names of all the memes
print(meme_names)

# Close the web driver
driver.quit()

