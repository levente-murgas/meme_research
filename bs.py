import requests
import random
import time
from bs4 import BeautifulSoup

# set headers to mimic browser behavior
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

with open('memes2.txt', 'w') as f:
    url = "https://knowyourmeme.com/memes/all"
    # make a request to the website with headers
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        photo_links = soup.find_all("a", class_="photo")
        for link in photo_links:
            f.write(link["href"] + "\n")

    cnt = 0
    for i in range(2, 1632):
        random_number = random.randint(2, 10)
        time.sleep(random_number)
        url = f"https://knowyourmeme.com/memes/all/page/{i}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            photo_links = soup.find_all("a", class_="photo")
            for link in photo_links:
                f.write(link["href"] + "\n")
            cnt += len(photo_links)
            print(f"Page {i} done. {cnt} links written so far.")

print(cnt)
