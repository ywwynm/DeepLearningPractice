from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import keras_dataset as flowers17


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


trn_x, trn_y = flowers17.get_images_and_labels('trn1')
val_x, val_y = flowers17.get_images_and_labels('val1')
tst_x, tst_y = flowers17.get_images_and_labels('tst1')

trn_data_gen = ImageDataGenerator(rescale=1./255)
trn_data_gen.fit(trn_x)
val_data_gen = ImageDataGenerator(rescale=1./255)
val_data_gen.fit(val_x)
tst_data_gen = ImageDataGenerator(rescale=1./255)
tst_data_gen.fit(tst_x)

trn_gen = trn_data_gen.flow(trn_x, trn_y, batch_size=64)
val_gen = val_data_gen.flow(val_x, val_y, batch_size=64)

model = alex_net()
model.fit_generator(trn_gen, epochs=200, validation_data=val_gen)
model.evaluate_generator(tst_data_gen)