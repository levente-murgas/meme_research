import requests
import random
import time
import pandas as pd
import re
from bs4 import BeautifulSoup

# set headers to mimic browser behavior
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

memes = []

def extract_name(soup: BeautifulSoup) -> str:
    return soup.find("h1").text.strip()

def extract_year(soup: BeautifulSoup) -> str:
    target_dt = soup.find('dt', text=re.compile(r'Year'))
    if target_dt is not None:
        target_dd = target_dt.find_next_sibling('dd')
        if target_dd is not None:
            return target_dd.text.strip()
        
    return "NaN"

def extract_type(soup: BeautifulSoup) -> str:
    target_dt = soup.find('dt', text=re.compile(r'Type'))
    if target_dt is not None:
        target_dd = target_dt.find_next_sibling('dd')
        if target_dd is not None:
            return target_dd.text.strip()
        
    return "NaN"

def extract_origin(soup: BeautifulSoup) -> str:
    target_dt = soup.find('dt', text=re.compile(r'Origin'))
    if target_dt is not None:
        target_dd = target_dt.find_next_sibling('dd')
        if target_dd is not None:
            return target_dd.text.strip()
        
    return "NaN"

def extract_tags(soup: BeautifulSoup) -> str:
    target_dt = soup.find('dt', text=re.compile(r'Tags'))
    if target_dt is not None:
        target_dd = target_dt.find_next_sibling('dd')
        if target_dd is not None:
            return target_dd.text.strip()
        
    return "NaN"

def extract_about(soup: BeautifulSoup) -> str:
    about = ""
    target_h2 = soup.find('h2', id = re.compile('about'))
    if target_h2 is not None:
        next_h2 = target_h2.find_next_sibling('h2')
        p_elements = []
        sibling = target_h2.find_next_sibling()
        while sibling is not None and sibling != next_h2:
            if sibling.name == 'p':
                p_elements.append(sibling)
            sibling = sibling.find_next_sibling()
        for p in p_elements:
            about += p.text
        return about.strip()
    
    return "NaN"

def extract_origin_article(soup: BeautifulSoup) -> str:
    origin_article = ""
    target_h2 = soup.find('h2', id = re.compile('origin'))
    if target_h2 is not None:
        next_h2 = target_h2.find_next_sibling('h2')
        p_elements = []
        sibling = target_h2.find_next_sibling()
        while sibling is not None and sibling != next_h2:
            if sibling.name == 'p':
                p_elements.append(sibling)
            sibling = sibling.find_next_sibling()
        for p in p_elements:
            origin_article += p.text
        return origin_article.strip()
    
    return "NaN"

def extract_spread(soup: BeautifulSoup) -> str:
    spread = ""
    target_h2 = soup.find('h2', id = re.compile('spread'))
    if target_h2 is not None:
        next_h2 = target_h2.find_next_sibling('h2')
        p_elements = []
        sibling = target_h2.find_next_sibling()
        while sibling is not None and sibling != next_h2:
            if sibling.name == 'p':
                p_elements.append(sibling)
            sibling = sibling.find_next_sibling()
        for p in p_elements:
            spread += p.text
        return spread.strip()
    
    return "NaN"

def extract_ref_image(soup: BeautifulSoup,url: str) -> str:
    name = url.split('/')[-1]
    target_img = soup.find('a', class_ = re.compile('photo left wide'))
    if target_img is not None:
        image_url = target_img['href']
        image_url = image_url.replace('masonry', 'original')
        format = image_url.split('.')[-1]
        response = requests.get(image_url)

        if response.status_code == 200:
            with open(f"imgs\{name}_ref.{format}", "wb") as f:
                f.write(response.content)
                print("Image downloaded successfully!")
        else:
            print("Error downloading image")
        return f"{name}_ref.{format}"
    
    return "NaN"

def extract_all_images(url: str) -> list:
    name = url.split('/')[-1]
    ## Get all the photos related to the meme
    url += "/photos"
    images = []
    page_cnt = 1
    found = True
    while found:
        found = False
        random_number = random.randint(2, 10)
        time.sleep(random_number)
        newurl = url + "/page/" + str(page_cnt)
        response = requests.get(newurl, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            target_divs = soup.find_all('div', class_= 'item')
            for div in target_divs:
                src = div.find('img')['data-src']
                found = True
                # If src ends with .jpg, .jpeg or .png, add it to the list
                if src.endswith('.jpg') or src.endswith('.png') or src.endswith('.jpeg'):
                    format = src.split('.')[-1]
                    src = src.replace('masonry', 'original')
                    response = requests.get(src, headers=headers)
                    if response.status_code == 200:
                        with open(f"imgs\{name}_{len(images)}.{format}", "wb") as f:
                            f.write(response.content)
                            images.append(f"{name}_{len(images)}.{format}")
                            print("Image downloaded successfully!")
                    else:
                        print("Error downloading image")
        page_cnt += 1

    return images

def extract_data(soup: BeautifulSoup,url: str) -> dict:
    name = extract_name(soup)
    year = extract_year(soup)
    type_of_meme = extract_type(soup)
    origin = extract_origin(soup)
    tags = extract_tags(soup)
    about = extract_about(soup)
    origin_article = extract_origin_article(soup)
    spread = extract_spread(soup)
    ref_image = extract_ref_image(soup,url)
    images = extract_all_images(url)
    meme = {
        "name": name,
        "year": year,
        "type": type_of_meme,
        "origin": origin,
        "tags": tags,
        "about": about,
        "origin_article": origin_article,
        "spread": spread,
        "ref_image": ref_image,
        "related_images": images,
        "example_url": url
    }
    return meme

with open('memes.txt', 'r') as f:
    meme_urls = f.readlines()
    for meme in meme_urls:
        random_number = random.randint(2, 10)
        time.sleep(random_number)
        url = "https://knowyourmeme.com" + meme.strip()
        # make a request to the website with headers
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            meme = extract_data(soup, url)
            memes.append(meme)
        else:
            print(f"Error: {response.status_code}")

df = pd.DataFrame.from_dict(memes)
df.to_csv('memes.csv')
