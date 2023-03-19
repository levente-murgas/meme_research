import requests
import random
import time
from bs4 import BeautifulSoup

# set headers to mimic browser behavior
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

with open('memes3.txt', 'a') as f:
    cnt = 0
    page_cnt = 1697
    found = True
    while found:
        found = False
        random_number = random.randint(2, 10)
        time.sleep(random_number)
        url = f"https://knowyourmeme.com/memes/all/page/{page_cnt}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            photo_links = soup.select('a.photo:not([class*=\'photo \'])')
            for link in photo_links:
                found = True
                f.write(link["href"] + "\n")
            cnt += len(photo_links)
            print(f"Page {page_cnt} done. {cnt} links written so far.")
        page_cnt += 1

print(cnt)
