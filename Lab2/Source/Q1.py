import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import boston_housing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import r2_score

# fix random seed for reproducibility
seed = 5
np.random.seed(seed)

# Load data
(train_x, train_y), (test_x, test_y) = boston_housing.load_data()
print('The data shape of boston housing:')
print('train_x: ', train_x.shape)
print('test_x: ', test_x.shape)
print('train_y: ', train_y.shape)
print('test_y: ', test_y.shape)

# process the data (Standardization)
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.fit_transform(test_x)

# build the model
activation_func = 'relu'
# activation_func = 'linear'
model = Sequential()
model.add(Dense(64, activation=activation_func, input_dim=13))
model.add(Dense(16, activation=activation_func))
model.add(Dense(1))

lrate = 0.001
adam = Adam(lr=lrate)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
# model.compile(loss='mean_squared_error',optimizer='RMSProp',metrics=['mae'])
print(model.summary())

# Training the model
batch_size = 10
epochs = 10
# tensorboard --logdir=Q1_logs
tbCallBack = TensorBoard(log_dir='Q1_logs')
history = model.fit(train_x, train_y,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(test_x, test_y),
                    callbacks=[tbCallBack])

# Print the result
loss, acc = model.evaluate(test_x, test_y, verbose=2, batch_size=batch_size)
print("Loss:", loss)

# Predict the result
pred_test_y = model.predict(test_x)
predict_accuracy = r2_score(test_y, pred_test_y)
print('pred_acc:', predict_accuracy)

# Plot loss
plt.figure(figsize=(4, 4))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Train and validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
# plt.savefig('./Q1/default.jpg')
plt.show()

# Plot loss with different parameter
default_image = mpimg.imread('./Q1/default.jpg')
lr0005 = mpimg.imread('./Q1/lr0005.jpg')
batch25 = mpimg.imread('./Q1/batch25.jpg')
RMSProp = mpimg.imread('./Q1/RMSProp.jpg')
linearFunc = mpimg.imread('./Q1/linearFunc.jpg')

# Plot lrate = 0.001 VS lrate = 0.005
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(default_image)
plt.title('lrate = 0.001')
plt.subplot(1, 2, 2)
plt.imshow(lr0005)
plt.title('lrate = 0.005')
plt.show()

# Plot batch_size = 10 VS batch_size = 25
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(default_image)
plt.title("batch_size = 10")
plt.subplot(1, 2, 2)
plt.imshow(batch25)
plt.title("batch_size = 25")
plt.show()

# Plot "optimizer=adam" VS "optimizer='RMSProp'"
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.imshow(default_image)
plt.title("optimizer=adam")
plt.subplot(1, 2, 2)
plt.imshow(RMSProp)
plt.title("optimizer='RMSProp'")
plt.show()

# Plot "optimizer=adam" VS "optimizer='RMSProp'"
plt.figure(figsize=(8, 8))
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(default_image)
plt.title("activation = 'relu'")
plt.subplot(1, 2, 2)
plt.imshow(linearFunc)
plt.title("activation = 'linear'")
plt.show()
