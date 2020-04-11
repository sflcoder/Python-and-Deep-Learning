from keras.layers import Embedding, Flatten
from keras.models import Sequential
from keras import layers
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('imdb_master.csv', encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values
print(y.shape)
# (100000,)
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))
# {'neg': 25000, 'pos': 25000, 'unsup': 50000}

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
print('input_dim: ', dimData)
# input_dim:  2000

# creating network
np.random.seed(1)
model = Sequential()
# Add embedding layer to the model, did you experience any improvement?
model.add(Embedding(vocab_size, 50, input_length=dimData))
model.add(Flatten())
model.add(layers.Dense(300, input_dim=dimData, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=5, verbose=True, validation_data=(X_test, y_test),
                    batch_size=256)

# Print the evaluation result
[test_loss, test_acc] = model.evaluate(X_test, y_test)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
# Evaluation result on Test Data : Loss = 0.8526971311759949, accuracy = 0.5059199929237366
