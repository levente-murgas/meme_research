import requests
import random
import time
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

def extract_image_url(soup: BeautifulSoup) -> str:
    target_img = soup.find('a', class_ = re.compile('photo left wide'))
    if target_img is not None:
        return target_img['href']
    
    return "NaN"

def extract_all_images(url: str) -> list:
    ## Get all the photos related to the meme
    url += "/photos/page/"
    images = []
    for i in range(1, 10):
        newurl = url + str(i)
        response = requests.get(newurl, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            target_divs = soup.find_all('div', class_= 'item')
            for div in target_divs:
                src = div.find('img')['src'] + "\n"
                print(src)
                # If src ends with .jpg or .png, add it to the list
                # if src.endswith('.jpg') or src.endswith('.png'):
                images.append(src)
    # while 
    # target_img = soup.find('a', class_ = re.compile('photo left wide'))
    # if target_img is not None:
    #     images.append(target_img['href'])
    return images


with open('memes.txt', 'r') as f:
    memes = f.readlines()
    for meme in memes:
        url = "https://knowyourmeme.com" + meme.strip()
        # make a request to the website with headers
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            name = extract_name(soup)
            year = extract_year(soup)
            type_of_meme = extract_type(soup)
            origin = extract_origin(soup)
            tags = extract_tags(soup)
            about = extract_about(soup)
            origin_article = extract_origin_article(soup)
            spread = extract_spread(soup)
            image_url = extract_image_url(soup)
            ## Not working yet
            # images = extract_all_images(url)



        else:
            print(f"Error: {response.status_code}")

