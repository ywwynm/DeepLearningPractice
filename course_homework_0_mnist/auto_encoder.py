import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


'''
Get encoded data "compressed" by trained AutoEncoder.
The test_data should be in shape [?, 784].
'''
def get_encoded_data(mnist, test_data):

  def encoder(input):
    l1 = tf.nn.sigmoid(tf.add(tf.matmul(input, weights['encoder_1']), biases['encoder_1']))
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(l1, weights['encoder_2']), biases['encoder_2']))
    return l2

  def decoder(hidden_layer):
    l2 = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, weights['decoder_2']), biases['decoder_2']))
    output = tf.nn.sigmoid(tf.add(tf.matmul(l2, weights['decoder_1']), biases['decoder_1']))
    return output

  learning_rate = 0.001
  batch_size = 128
  epoches = 50

  input_size = 784
  X = tf.placeholder("float", [None, input_size])

  hidden_1_size = 256
  hidden_2_size = 128

  weights = {
    'encoder_1': tf.Variable(tf.random_normal([input_size, hidden_1_size])),
    'encoder_2': tf.Variable(tf.random_normal([hidden_1_size, hidden_2_size])),

    'decoder_2': tf.Variable(tf.random_normal([hidden_2_size, hidden_1_size])),
    'decoder_1': tf.Variable(tf.random_normal([hidden_1_size, input_size]))
  }

  biases = {
    'encoder_1': tf.Variable(tf.random_normal([hidden_1_size])),
    'encoder_2': tf.Variable(tf.random_normal([hidden_2_size])),

    'decoder_2': tf.Variable(tf.random_normal([hidden_1_size])),
    'decoder_1': tf.Variable(tf.random_normal([input_size]))

  }

  hidden_layer = encoder(X)
  output = decoder(hidden_layer)

  y_true = X
  y_pred = output

  loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    total_batch_count = int(mnist.train.num_examples / batch_size)
    print("Start to train AutoEncoder --------------------------------")
    for epoch in range(epoches):
      for i in range(total_batch_count):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
        if i % 10 == 0:
          print('epoch: ', epoch + 1, ", i: ", i, ", loss: ", l)

    print("end training AutoEncoder --------------------------------")
    encode_result_test = sess.run(y_pred, feed_dict={X: test_data})
    return encode_result_test

    #
    # print(encode_result_test)
    #
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(10):
    #   a[0][i].imshow(np.reshape(mnist.test.images[i + 10], (28, 28)))
    #   a[1][i].imshow(np.reshape(encode_result_test[i + 10], (28, 28)))
    # plt.show()