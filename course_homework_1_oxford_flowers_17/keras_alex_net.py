from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


def alex_net():
  model = Sequential()

  model.add(Conv2D(input_shape=(224, 224, 3), filters=96, kernel_size=11, activation='relu'))
  model.add(MaxPool2D(pool_size=3, strides=2))

  model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
  model.add(MaxPool2D(pool_size=3, strides=2))

  model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=384, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=384, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(pool_size=3, strides=2))

  model.add(Flatten())
  model.add(Dense(units=4096, activation='tanh'))
  model.add(Dropout(0.5))
  model.add(Dense(units=4096, activation='tanh'))
  model.add(Dropout(0.5))
  model.add(Dense(units=17, activation='softmax'))

  return model

