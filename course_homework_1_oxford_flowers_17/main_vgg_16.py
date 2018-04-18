from tensorflow.python.keras.optimizers import RMSprop

import course_homework_1_oxford_flowers_17.dataset as dataset
import course_homework_1_oxford_flowers_17.vgg_16 as vgg_16

learning_rate = 0.001
input_width = input_height = 224
channel = 3
output_size = 17
batch_size = 32
epochs = 500

train_set_1 = dataset.get_train_set(1)
test_set_1 = dataset.get_test_set(1)

train_set_1 = train_set_1.shuffle(buffer_size=10000)
train_set_1 = train_set_1.repeat(epochs)
train_set_1 = train_set_1.batch(batch_size)

model = vgg_16.vgg_16()
rmsprop = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
