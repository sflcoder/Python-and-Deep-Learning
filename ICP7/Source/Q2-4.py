from bs4 import BeautifulSoup
import urllib.request


def webScraping(url):
    f = open("input.txt", 'w+', encoding='utf-8')
    source_code = urllib.request.urlopen(url)
    plain_text = source_code
    soup = BeautifulSoup(plain_text, "html.parser")
    f.write(soup.body.text.encode('utf-8', "rb").decode('utf-8'))


webScraping("https://en.wikipedia.org/wiki/Google")

file = open("input.txt", "rb")
originalText = file.read().decode()

# Tokenization
print("==================Tokenization==================")
import nltk

wTokens = nltk.word_tokenize(originalText)
for wToken in wTokens:
    print(wToken)

sTokens = nltk.sent_tokenize(originalText)
for sToken in sTokens:
    print(sToken)

print("==================POS==================")
wTokens = nltk.word_tokenize(originalText)
print(nltk.pos_tag(wTokens))

# Stemming
print("==================Stemming==================")
from nltk.stem import PorterStemmer

pStremmer = PorterStemmer()
for wToken in wTokens:
    print(pStremmer.stem(wToken))

# Lemmatization
print("==================Lemmatization==================")
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
for wToken in wTokens:
    print(lemmatizer.lemmatize(wToken))

# Trigram
print("==================Trigram==================")
from nltk.util import ngrams

n = 3
trigrams = list(ngrams(originalText.split(), n))
for grams in trigrams:
    print(grams)

# Named Entity Recognition
print("==================Named Entity Recognition==================")
from nltk import wordpunct_tokenize, pos_tag, ne_chunk

print(ne_chunk(pos_tag(wordpunct_tokenize(originalText))))
