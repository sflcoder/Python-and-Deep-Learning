import nltk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter

# Read the data from nlp_input.txt and then close the resource.
inputFile = open("nlp_input.txt", "rb")
originalText = inputFile.read().decode()
inputFile.close()

# Tokenize the words in nlp_input.txt
wordTokens = nltk.word_tokenize(originalText)
print("\nWord tokens in nlp_input.txt:\n", wordTokens)

# Lemmatize all word tokens
lemmatizer = WordNetLemmatizer()
lemmatized_wordTokens = ', '.join([lemmatizer.lemmatize(wordToken) for wordToken in wordTokens])
print("\nLemmatized word tokens:\n", lemmatized_wordTokens)

# Find trigrams and save them to a list
n = 3
trigrams = list(ngrams(wordTokens, n))
print("\nAll trigrams:\n", trigrams)

# Count the frequency of trigrams and extract the top 10 repeated trigrams based on the frequency.
top10Trigrams = Counter(trigrams).most_common(10)
print("\nTop 10 trigrams with frequency:\n", top10Trigrams)

# Remove the frequency and save the top 10 trigrams to a list
top10TrigramsStrs = []
for trigram in top10Trigrams:
    str = ' '.join(trigram[0])
    top10TrigramsStrs.append(str)
print("\nTop 10 trigrams:\n", top10TrigramsStrs)

# Concatenate the sentence with the top 10 trigrams
concatenatedStr = ''
sentTokens = nltk.sent_tokenize(originalText)
for sToken in sentTokens:
    for top10TrigramsStr in top10TrigramsStrs:
        if (top10TrigramsStr in sToken):
            concatenatedStr += sToken
            break

# print the result
print("\nConcatenated sentence with the top 10 trigrams:\n", concatenatedStr)
