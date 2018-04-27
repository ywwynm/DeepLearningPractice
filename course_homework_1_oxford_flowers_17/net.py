import tensorflow as tf
import numpy as np
import model


def conv_2d(id, input, filter_size, in_channel, out_channel, strides=1, padding='SAME'):
  # filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, out_channel]) * 1e-4)

  # input_size = 1.0
  # for dim in shape[:-1]:
  #   input_size *= float(dim)
  # max_val = math.sqrt(3 / (filter_size * filter_size * in_channel)) * 1.0
  # filter = tf.Variable(tf.random_uniform([filter_size, filter_size, in_channel, out_channel], -max_val, max_val))

  # filter = tf.Variable(tf.truncated_normal([filter_size, filter_size, in_channel, out_channel], stddev=0.1))

  filter = tf.get_variable("W_" + str(id), [filter_size, filter_size, in_channel, out_channel],
                           initializer=tf.contrib.layers.xavier_initializer())

  conv = tf.nn.conv2d(input, filter, [1, strides, strides, 1], padding)
  b = tf.Variable(tf.zeros([out_channel]))

  # b = tf.get_variable("b_" + str(id), [out_channel],
  #                 initializer=tf.contrib.layers.xavier_initializer())
  added = tf.nn.bias_add(conv, b)
  bn = tf.layers.batch_normalization(added, training=model.flag_training)
  return tf.nn.relu(bn)


def max_pool(input, ksize, strides, padding='SAME'):
  return tf.nn.max_pool(input, [1, ksize, ksize, 1], [1, strides, strides, 1], padding)


def lrn(input, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75):
  return tf.nn.lrn(input, depth_radius, bias, alpha, beta)


def fully_connected(id, input, n_units, activation='relu', keep_prob=0.5):
  input_size = np.prod(input.get_shape().as_list()[1:])
  flattened = tf.reshape(input, [-1, input_size])
  # W = tf.Variable(tf.truncated_normal([input_size, n_units]) * 0.001)
  # W = tf.Variable(tf.truncated_normal([input_size, n_units], stddev=0.2))

  W = tf.get_variable("W_" + str(id), [input_size, n_units],
                      initializer=tf.contrib.layers.xavier_initializer())
  b = tf.Variable(tf.zeros([n_units]))

  # b = tf.get_variable("b_" + str(id), [n_units],
  #                 initializer=tf.contrib.layers.xavier_initializer())
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