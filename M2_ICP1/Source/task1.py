from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# load dataset from diabetes.csv
dataset = pd.read_csv("diabetes.csv", header=None).values
# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8], test_size=0.25, random_state=87)
np.random.seed(155)

# create model
my_second_nn = Sequential()
# hidden layer
my_second_nn.add(Dense(20, input_dim=8, activation='relu'))
# add the second hidden layer
my_second_nn.add(Dense(14, activation='relu'))
# add the third more hidden layer
my_second_nn.add(Dense(7, activation='relu'))
# output layer
my_second_nn.add(Dense(1, activation='sigmoid'))
my_second_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_second_nn_fitted = my_second_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)

# Print the result
print(my_second_nn.summary())
result = my_second_nn.evaluate(X_test, Y_test)
print(result)

print('\n\n The result with three hiden layers:')
print(' loss: ', result[0], '\n Accuracy: ', result[1])