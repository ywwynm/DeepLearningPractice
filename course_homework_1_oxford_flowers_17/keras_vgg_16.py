from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout


def vgg_16():
  model = Sequential()

  model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(strides=2))

  model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(strides=2))

  model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(strides=2))

  model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(strides=2))

  model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
  model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
  model.add(MaxPool2D(strides=2))

  model.add(Flatten())
  model.add(Dense(units=4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(units=4096, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(units=17, activation='softmax'))

  return model