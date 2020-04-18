# Simple CNN model for CIFAR-10
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from matplotlib import pyplot as plt

# K.set_image_dim_ordering('th')
K.image_data_format()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(y_test)

labels = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Load saved model
model = load_model('task1_model.h5')

# predict the first 4 image of the test data. Then, print the actual label for those 4 images (label means the probability associated with them)
# to check if the model predicted correctly or not
predicted_labels = list()
labels_probability = list()

# predit the first 4 image of the test data with the saved model
for i in range(4):
    predicted = model.predict(X_test[i].reshape(-1, 32, 32, 3))
    predicted_labels.append(labels[predicted.argmax()])
    labels_probability.append(labels[predicted.argmax()] + ': ' + str(np.max(predicted)))


print('\nThe predicted probability for the first 4 image of the test data:')
for i in range(len(labels_probability)):
    print("Index", i, labels_probability[i])

print('\nThe predicted label for the first 4 image of the test data:\n', predicted_labels)

# print the actual label for the first 4 image of the test data
actual_labels = list()
for i in range(4):
    actual_labels.append(labels[np.argmax(y_test[i])])
print('\nThe actual label for the first 4 image of the test data:\n',actual_labels)


# draw the first 4 image of the test data
for i in range(4):
    plt.subplot(140 + 1 + i)
    plt.title(actual_labels[i])
    plt.imshow(X_test[i])
plt.show()
