import requests
from bs4 import BeautifulSoup

url = "https://catalog.umkc.edu/course-offerings/graduate/comp-sci"
page = requests.get(url)    # Getting the html page content

soup = BeautifulSoup(page.content, 'html.parser')   # Parsing page content as HTML

# Finding the required information based on the HTML tag and its class
for x, y in zip(soup.find_all("span", class_="title"), soup.find_all("p", class_="courseblockdesc")):
    print(x.string, y.string)
    print()