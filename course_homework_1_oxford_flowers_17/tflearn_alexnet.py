""" AlexNet.
Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.
References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.
Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17
X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# Building 'AlexNet'
network = input_data(shape=[None, 224, 224, 3])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 17, activation='softmax')
network = regression(network, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
# model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
#           show_metric=True, batch_size=64, snapshot_step=200,
#           snapshot_epoch=False, run_id='alexnet_oxflowers17')


import course_homework_1_oxford_flowers_17.dataset as dataset
train_set_1 = dataset.get_train_set(1)
train_set_1 = train_set_1.shuffle(buffer_size=10000)
train_set_1 = train_set_1.batch(64)

itr_trn_1 = train_set_1.make_initializable_iterator()
next_element_trn_1 = itr_trn_1.get_next()

import tensorflow as tf
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  print("Start to train classifier")
  for epoch in range(1000):
    print("------------------ Epoch %d starts ------------------" % (epoch + 1))
    sess.run(itr_trn_1.initializer)
    batch_idx = 1
    while True:
      try:
        X_batch, Y_batch = sess.run(next_element_trn_1)
        model.fit(X_batch, Y_batch, n_epoch=1)
        batch_idx += 1
      except tf.errors.OutOfRangeError:  # this epoch ends
        break

  #   if (epoch + 1) % 40 == 0:
  #     print("------------------ evaluating accuracy on validation set ------------------")
  #     itr_val_1 = validation_set_1.make_one_shot_iterator()
  #     next_element_val_1 = itr_val_1.get_next()
  #     pred_vals = []
  #     while True:
  #       try:
  #         X_batch_tst, Y_batch_tst = sess.run(next_element_val_1)
  #         pred = tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32)
  #         pred_val = sess.run(pred, feed_dict={X: X_batch_tst, y_true: Y_batch_tst})
  #         pred_vals.append(pred_val)
  #       except tf.errors.OutOfRangeError:
  #         break
  #     print("accuracy: %f" % (np.mean(pred_vals)))
  #
  # print("Classifier has been trained, total time: %f" % (time.time() - start_train_time))