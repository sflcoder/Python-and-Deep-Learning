from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt

# fix random seed for reproducibility
seed = 20
np.random.seed(seed)

# this is the size of our encoded representations
encoding_dim = 32

# this is our input placeholder
input_img = Input(shape=(784,))

# "encoded" is the encoded representation of the input
hidden_encoder = Dense(98, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='relu')(hidden_encoder)

# "decoded" is the lossy reconstruction of the input
hidden_decoder = Dense(98, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(hidden_decoder)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# build the encoder
encoder = Model(input_img, encoded)

# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = autoencoder.fit(x_train, x_train,
                          epochs=14,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test, x_test),
                          verbose=2)

# plot the Original image and Autoencoder output
plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title('Original image')
plt.subplot(1, 2, 2)
prediction = autoencoder.predict(x_test)
plt.imshow(prediction[0].reshape(28, 28), cmap='gray')
plt.title('Autoencoder output')
plt.show()

# plot the encoded image(Compressed image)
encoded_image = encoder.predict(x_test)
print(encoded_image[0].shape)
plt.imshow(encoded_image[0].reshape(8, 4), cmap='gray')
plt.title('After encoding')
plt.show()

# Plot the loss and accuracy using history object
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Train and validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()
