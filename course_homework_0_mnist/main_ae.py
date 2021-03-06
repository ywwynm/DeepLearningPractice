import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import course_homework_0_mnist.auto_encoder as ae

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

encoder = ae.get_trained_encoder(mnist)

learning_rate = 0.001
batch_size = 128
epoches = 1000

input_size = 256
output_size = 10

X = tf.placeholder("float", [None, input_size])

W = tf.Variable(tf.zeros([input_size, output_size]))
b = tf.Variable(tf.zeros([output_size]))

y_pred = tf.nn.softmax(tf.add(tf.matmul(X, W), b))
y_true = tf.placeholder("float", [None, output_size])

loss = -tf.reduce_sum(y_true * tf.log(y_pred))  # cross entropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  times = []
  losses = []
  error_rates = []
  encoded_test_images = sess.run(encoder(mnist.test.images))

  print("Start to train classifier --------------------")
  for i in range(epoches):
    batch_x_ori, batch_y = mnist.train.next_batch(batch_size)
    batch_x = sess.run(encoder(batch_x_ori))  # transform a tf.Tensor to numpy.NdArray
    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, y_true: batch_y})
    if i % 10 == 0:
      times.append(i)
      losses.append(l)
      pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
      accuracy = tf.reduce_mean(tf.cast(pred, "float"))
      accuracy_val = sess.run(accuracy, feed_dict={X: encoded_test_images, y_true: mnist.test.labels})
      error_rates.append(1 - accuracy_val)
      print("i: ", i, ", loss: ", l, ", error rate: ", 1 - accuracy_val)
  print("Classifier has been trained --------------------")

  plt.plot(times, losses)
  plt.xlabel('trained times')
  plt.ylabel('cross entropy')
  plt.show()

  plt.plot(times, error_rates)
  plt.xlabel('trained times')
  plt.ylabel('error rate')
  plt.show()

  # pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
  # accuracy = tf.reduce_mean(tf.cast(pred, "float"))
  # print("accuracy", sess.run(accuracy, feed_dict={X: encoded_test_images, y_true: mnist.test.labels}))
