from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# load dataset
breastcancer_dataset = pd.read_csv("breastcancer.csv")

# Encoding a categorical feature: Converting the value of ['diagnosis'] from 'M' to 0, 'B' to 1
diagnosis_mapping = {'M': 0, 'B': 1}
breastcancer_dataset['diagnosis'] = breastcancer_dataset['diagnosis'].map(diagnosis_mapping)
X_train, X_test, Y_train, Y_test = train_test_split(breastcancer_dataset.iloc[:, 2:32], breastcancer_dataset.iloc[:, 1],
                                                    test_size=0.25, random_state=87)

# Normalize the data and then feed it to the model
sc = StandardScaler()
sc.fit(breastcancer_dataset.iloc[:, 2:32])
X_scaled_array = sc.transform(breastcancer_dataset.iloc[:, 2:32])
X_scaled = pd.DataFrame(X_scaled_array, columns=breastcancer_dataset.iloc[:, 2:32].columns)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, breastcancer_dataset.iloc[:, 1], test_size=0.25,
                                                    random_state=0)

np.random.seed(155)
# create model
model = Sequential()
# hidden layer
model.add(Dense(20, input_dim=30, activation='relu'))
# output layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model2_fitted = model.fit(X_train, Y_train, epochs=100, initial_epoch=0)

# Print the result
print(model.summary())
result = model.evaluate(X_test, Y_test)
print(result)
print('\n loss: ', result[0], '\n Accuracy: ', result[1])
