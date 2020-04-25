import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
import numpy as np

from sklearn.preprocessing import LabelEncoder

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

data = pd.read_csv('spam.csv', encoding='latin-1')
# Keeping only the neccessary columns
data = data[['v1', 'v2']]

print(data['v1'][0])
data['v2'] = data['v2'].apply(lambda x: x.lower())
data['v2'] = data['v2'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))
print(type(data['v1']))
print(type(data['v2']))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['v2'].values)
X = tokenizer.texts_to_sequences(data['v2'].values)
print(X)
X = pad_sequences(X)
print(X)
print(type(X))

embed_dim = 128
lstm_out = 196


def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


# model = KerasClassifier(build_fn=createmodel,verbose=0)

labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['v1'])
print('========================================')
unique, counts = np.unique(data['v1'], return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(integer_encoded, return_counts=True)
print(dict(zip(unique, counts)))

y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

batch_size = 32
model = createmodel()
print(type(X_train))
model.fit(X_train, Y_train, epochs=7, batch_size=batch_size, verbose=2)

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)
print("Loss:", score)
print("Accuracy: %.2f%%" % (acc * 100))

