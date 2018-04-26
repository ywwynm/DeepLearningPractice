import tensorflow as tf
import numpy as np
import math

def conv_2d(num, input, filter_size, in_channel, out_channel, strides=1, padding='SAME'):
  # filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, out_channel]) * 1e-4)

  # input_size = 1.0
  # for dim in shape[:-1]:
  #   input_size *= float(dim)
  max_val = math.sqrt(3 / (filter_size * filter_size * in_channel)) * 1.0
  filter = tf.Variable(tf.random_uniform([filter_size, filter_size, in_channel, out_channel], -max_val, max_val))

  # filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, out_channel], stddev=0.1))

  filter = tf.get_variable("W_" + str(num), [filter_size, filter_size, in_channel, out_channel],
                           initializer=tf.contrib.layers.xavier_initializer())

  conv = tf.nn.conv2d(input, filter, [1, strides, strides, 1], padding)
  b = tf.Variable(tf.zeros([out_channel]))
  added = tf.nn.bias_add(conv, b)
  return tf.nn.relu(added)

def max_pool(input, ksize, strides, padding='SAME'):
  return tf.nn.max_pool(input, [1, ksize, ksize, 1], [1, strides, strides, 1], padding)

def lrn(input, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75):
  return tf.nn.lrn(input, depth_radius, bias, alpha, beta)

def fully_connected(num, input, n_units, activation='relu', keep_prob=0.5):
  input_size = np.prod(input.get_shape().as_list()[1:])
  flattened = tf.reshape(input, [-1, input_size])
  # W = tf.Variable(tf.truncated_normal([input_size, n_units]) * 0.001)
  # W = tf.Variable(tf.truncated_normal([input_size, n_units], stddev=0.2))
  W = tf.get_variable("W_" + str(num), [input_size, n_units],
                           initializer=tf.contrib.layers.xavier_initializer())
  b = tf.Variable(tf.zeros([n_units]))
  added = tf.nn.xw_plus_b(flattened, W, b)
  if activation == 'relu':
    activated = tf.nn.relu(added)
  elif activation == 'tanh':
    activated = tf.nn.tanh(added)
  elif activation == 'softmax':
    activated = tf.nn.softmax(added)
  else:
    activated = added
  if keep_prob > 0:
    return tf.nn.dropout(activated, keep_prob=keep_prob)
  else:
    return activated


def aein_net(input):
  net = conv_2d(1, input, filter_size=5, in_channel=3, out_channel=64, strides=4, padding='SAME')
  net = max_pool(net, ksize=3, strides=2, padding='SAME')
  net = lrn(net)

  net = conv_2d(2, net, filter_size=4, in_channel=64, out_channel=128, padding='SAME')
  net = lrn(net)
  net = max_pool(net, ksize=3, strides=2, padding='SAME')

  net = conv_2d(3, net, filter_size=4, in_channel=128, out_channel=256, padding='SAME')
  # net = conv_2d(net, filter_size=4, in_channel=256, out_channel=384, padding='SAME')
  # net = max_pool(net, ksize=3, strides=2, padding='VALID')

  net = fully_connected(4, net, n_units=1024, activation="relu", keep_prob=0.5)
  net = fully_connected(5, net, n_units=2048, activation="relu", keep_prob=0.5)
  net = fully_connected(6, net, n_units=17, activation="", keep_prob=-1.0)
  return net