from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from matplotlib import pyplot as plt

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input

# Add one more hidden layer to autoencoder
# add one more encoder hidden layer
hidden_encoder = Dense(98, activation='relu')(input_img)
encoded = Dense(encoding_dim, activation='relu')(hidden_encoder)

# "decoded" is the lossy reconstruction of the input
# add one more decoder hidden layer
hidden_decoder = Dense(98, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(hidden_decoder)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# build the encoder model
encoder = Model(input_img, encoded)

# this model maps an input to its encoded representation
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

from keras.datasets import fashion_mnist

(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

history = autoencoder.fit(x_train_noisy, x_train,
                          epochs=10,
                          batch_size=256,
                          shuffle=True,
                          validation_data=(x_test_noisy, x_test))

# plot the Original image , image with noise and Autoencoder output
plt.figure(figsize=(6, 4))
plt.subplot(1, 3, 1)
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title('Original image')
plt.subplot(1, 3, 2)
plt.imshow(x_test_noisy[0].reshape(28, 28), cmap='gray')
plt.title('introducing noise')
plt.subplot(1, 3, 3)
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

print(history.history.keys())

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
