from sklearn.datasets import fetch_20newsgroups
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

# Apply the code on 20_newsgroup data set we worked in the previous classes
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True)
sentences = newsgroups_train.data
y = newsgroups_train.target

print(y.shape)

unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
max_review_len = max([len(s.split()) for s in sentences])
print('Max review length: ', max_review_len)
vocab_size = len(tokenizer.word_index) + 1
print('vocab_size: ', vocab_size)
# getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
padded_docs = pad_sequences(sentences, maxlen=max_review_len)

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

print('X_train.shape: ', X_train.shape)
# Number of features
# print(input_dim)
dimData = np.prod(X_train.shape[1:])
print('input_dim', dimData)

# creating network
np.random.seed(1)
model = Sequential()
model.add(layers.Dense(300, input_dim=dimData, activation='relu'))
model.add(layers.Dense(20, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test),
                    batch_size=256)

# Print the evaluation result
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


