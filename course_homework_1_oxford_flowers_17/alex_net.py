import tensorflow as tf
import numpy as np
import math

def conv_2d(input, filter_size, in_channel, out_channel, strides=1, padding='SAME'):
  # filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, out_channel]) * 1e-4)

  # input_size = 1.0
  # for dim in shape[:-1]:
  #   input_size *= float(dim)
  # max_val = math.sqrt(3 / (filter_size * filter_size * in_channel)) * 1.0
  # filter = tf.Variable(tf.random_uniform([filter_size, filter_size, in_channel, out_channel], -max_val, max_val))

  filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, out_channel], stddev=0.1))

  conv = tf.nn.conv2d(input, filter, [1, strides, strides, 1], padding)
  b = tf.Variable(tf.zeros([out_channel]))
  added = tf.nn.bias_add(conv, b)
  return tf.nn.relu(added)

def max_pool(input, ksize, strides, padding='VALID'):
  return tf.nn.max_pool(input, [1, ksize, ksize, 1], [1, strides, strides, 1], padding)

def lrn(input, depth_radius=2, bias=1.0, alpha=2e-5, beta=0.75):
  return tf.nn.lrn(input, depth_radius, bias, alpha, beta)

def fully_connected(input, n_units, activation='relu', keep_prob=0.5):
  input_size = np.prod(input.get_shape().as_list()[1:])
  flattened = tf.reshape(input, [-1, input_size])
  # W = tf.Variable(tf.truncated_normal([input_size, n_units]) * 0.001)
  W = tf.Variable(tf.truncated_normal([input_size, n_units], stddev=0.2))
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


def alex_net(input):
  net = conv_2d(input, filter_size=11, in_channel=3, out_channel=96, strides=4, padding='VALID')
  net = max_pool(net, ksize=3, strides=2, padding='VALID')
  net = lrn(net)
  net = conv_2d(net, filter_size=5, in_channel=96, out_channel=256, padding='SAME')
  net = max_pool(net, ksize=3, strides=2, padding='VALID')
  net = lrn(net)
  net = conv_2d(net, filter_size=3, in_channel=256, out_channel=384, padding='SAME')
  net = conv_2d(net, filter_size=3, in_channel=384, out_channel=384, padding='SAME')
  net = conv_2d(net, filter_size=3, in_channel=384, out_channel=256, padding='SAME')
  net = max_pool(net, ksize=3, strides=2, padding='VALID')
  # net = lrn(net)
  net = fully_connected(net, n_units=4096, activation='relu')
  net = fully_connected(net, n_units=4096, activation='relu')
  net = fully_connected(net, n_units=17, activation="", keep_prob=-1.0)
  return net