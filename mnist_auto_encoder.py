import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

learning_rate = 0.001
batch_size = 128
epoches = 10

input_size = 784
X = tf.placeholder("float", [None, input_size])

hidden_1_size = 256
hidden_2_size = 128
hidden_3_size = 64
hidden_4_size = 10

weights = {
  'encoder_1': tf.Variable(tf.random_normal([input_size, hidden_1_size])),
  'decoder_1': tf.Variable(tf.random_normal([hidden_1_size, input_size])),
  'encoder_2': tf.Variable(tf.random_normal([hidden_1_size, hidden_2_size])),
  'decoder_2': tf.Variable(tf.random_normal([hidden_2_size, hidden_1_size])),
  'encoder_3': tf.Variable(tf.random_normal([hidden_2_size, hidden_3_size])),
  'decoder_3': tf.Variable(tf.random_normal([hidden_3_size, hidden_2_size])),
  'encoder_4': tf.Variable(tf.random_normal([hidden_3_size, hidden_4_size])),
  'decoder_4': tf.Variable(tf.random_normal([hidden_4_size, hidden_3_size]))
}

biases = {
  'encoder_1': tf.Variable(tf.random_normal([hidden_1_size])),
  'decoder_1': tf.Variable(tf.random_normal([input_size])),
  'encoder_2': tf.Variable(tf.random_normal([hidden_2_size])),
  'decoder_2': tf.Variable(tf.random_normal([hidden_1_size])),
  'encoder_3': tf.Variable(tf.random_normal([hidden_3_size])),
  'decoder_3': tf.Variable(tf.random_normal([hidden_2_size])),
  'encoder_4': tf.Variable(tf.random_normal([hidden_4_size])),
  'decoder_4': tf.Variable(tf.random_normal([hidden_3_size]))
}


def encoder(input):
  l1 = tf.nn.relu(tf.add(tf.matmul(input, weights['encoder_1']), biases['encoder_1']))
  l2 = tf.nn.relu(tf.add(tf.matmul(l1, weights['encoder_2']), biases['encoder_2']))
  l3 = tf.nn.relu(tf.add(tf.matmul(l2, weights['encoder_3']), biases['encoder_3']))
  l4 = tf.nn.relu(tf.add(tf.matmul(l3, weights['encoder_4']), biases['encoder_4']))
  return l4


def decoder(hidden_layer):
  l4 = tf.nn.relu(tf.add(tf.matmul(hidden_layer, weights['decoder_4']), biases['decoder_4']))
  l3 = tf.nn.relu(tf.add(tf.matmul(l4, weights['decoder_3']), biases['decoder_3']))
  l2 = tf.nn.relu(tf.add(tf.matmul(l3, weights['decoder_2']), biases['decoder_2']))
  output = tf.nn.relu(tf.add(tf.matmul(l2, weights['decoder_1']), biases['decoder_1']))
  return output


hidden_layer = encoder(X)
output = decoder(hidden_layer)

y_true = X
y_pred = output

loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  total_batch_count = int(mnist.train.num_examples / batch_size)
  for epoch in range(epoches):
    for i in range(total_batch_count):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
      if i % 10 == 0:
        print('epoch: ', epoch, ", i: ", i, ", loss: ", l)