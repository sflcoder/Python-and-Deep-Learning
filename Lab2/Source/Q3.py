import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import load_model


def read_data(path):
    images = []
    classes = []
    number = -1
    for root, folder, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            image = load_img(file_path)
            image = img_to_array(image.resize((32, 32)))
            images.append(image)
            classes.append([number])
        number += 1
    return images, classes


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

images_path = './natural-images/data/natural_images'

# Read all the classes names
classes_names = os.listdir(images_path)
print(classes_names)

# Read images and labels
images, labels = read_data(images_path)

# Count the number of images in each class
unique, counts = np.unique(labels, return_counts=True)
print('\nThe number of images in each class')
print(dict(zip(classes_names, counts)))

# plot the number of images in each class
plt.bar(classes_names, counts)
plt.xlabel('Classes')
plt.ylabel('The number of images')
plt.show()

# convert data to float and scale values between 0 and 1
images = np.array(images).astype('float')
images /= 255.0

# print(labels.shape)
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, random_state=42, test_size=.33)

# change the labels frominteger to one-hot encoding. to_categorical is doing the same thing as LabelEncoder()
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)
num_classes = train_labels_one_hot.shape[1]

# Create the model
model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
print('\ninput_shape: ', train_images.shape[1:])

# Convolutional input layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
# Dropout layer at 20%.
model.add(Conv2D(32, (3, 3), input_shape=train_images.shape[1:], padding='same', activation='relu',
                 kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

# Convolutional layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
# Max Pool layer with size 2×2.
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
# Dropout layer at 20%.
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

# Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
# Max Pool layer with size 2×2.
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional layer, 128 feature maps with a size of 3×3 and a rectifier activation function.
# Dropout layer at 20%.
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

# Convolutional layer,128 feature maps with a size of 3×3 and a rectifier activation function.
# Max Pool layer with size 2×2.
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer.
model.add(Flatten())

# Dropout layer at 20%.
model.add(Dropout(0.2))

# Fully connected layer with 1024 units and a rectifier activation function.
# Dropout layer at 20%.
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

# Fully connected layer with 512 units and a rectifier activation function.
# Dropout layer at 20%.
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))

# Fully connected output layer with 10 units and a Softmax activation function
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 15
lrate = 0.01
decay = lrate / epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
history = model.fit(train_images, train_labels_one_hot, validation_data=(test_images, test_labels_one_hot),
                    epochs=epochs, batch_size=32, verbose=2, )

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

# Print the evaluation result
[test_loss, test_acc] = model.evaluate(test_images, test_labels_one_hot, verbose=2, batch_size=32)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))

# # Save the model
# model.save('Q3_model.h5')
# # Load saved model
# model = load_model('Q3_model.h5')

# predit the image of the test data with the trained model
predicted_labels = model.predict_classes(test_images)
predicted_labels_names = list()
for i in range(2):
    predicted_labels_names.append(classes_names[predicted_labels[i]])

# plot the image and predicted label
plt.figure(figsize=(4, 2))
for i in range(2):
    plt.subplot(1, 2, 1 + i)
    plt.imshow(test_images[i])
    plt.title(predicted_labels_names[i])
plt.show()
