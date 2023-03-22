import random
import time
import requests
import numpy as np

# set headers to mimic browser behavior
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

while True:
    random_number = np.random.random(1)[0]
    time.sleep(random_number)
    url = f"https://knowyourmeme.com/memes/all"
    response = requests.get(url, headers=headers)
    print(response.status_code)