import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from keras.models import load_model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

data = pd.read_csv('sentiment-analysis-on-movie-reviews\\train.tsv', delimiter='\t')
print(data.shape)
# Keeping only the neccessary columns
data = data[['Phrase', 'Sentiment']]
data['Phrase'] = data['Phrase'].apply(lambda x: x.lower())
data['Phrase'] = data['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['Phrase'].values)
X = tokenizer.texts_to_sequences(data['Phrase'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196


def createmodel():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

sentiment_names = ['negative','somewhat negative','neutral','somewhat positive','positive']
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['Sentiment'])
# Count the number of element in each Sentiment
unique, counts = np.unique(data['Sentiment'], return_counts=True)
print('The number of each sentiment labels: ')
print(dict(zip(sentiment_names, counts)))

# plot the number of element in each sentiment
plt.figure(figsize=(8, 4))
plt.bar(sentiment_names, counts)
plt.xlabel('Sentiment')
plt.ylabel('The number of element in each sentiment')
plt.show()

y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)

batch_size = 32
model = createmodel()
history = model.fit(X_train, Y_train,
                    validation_data=(X_test, Y_test),
                    epochs=10,
                    batch_size=batch_size,
                    verbose=2)



# plot the loss for both training data and validation data.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss for both training data and validation data')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# plot the accuracy for both training data and validation data.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy for both training data and validation data')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

model.save('Q5_model10.h5')
model = load_model('Q5_model10.h5')
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

print("Loss:", score)
print("Accuracy: %.2f%%" % (acc * 100))

predicted_labels = model.predict_classes(X_test)
print('The predicted Sentiment label for X_test[0]:', sentiment_names[predicted_labels[0]])
print('The actual Sentiment label for X_test[0]: ',sentiment_names[np.argmax(Y_test[0])])
