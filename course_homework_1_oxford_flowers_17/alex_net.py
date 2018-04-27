import tensorflow as tf
import numpy as np
from net import conv_2d, max_pool, lrn, fully_connected


def alex_net(input):
  net = conv_2d(1, input, filter_size=11, in_channel=3, out_channel=96, strides=4, padding='SAME')
  net = max_pool(net, ksize=3, strides=2, padding='SAME')
  net = lrn(net)

  net = conv_2d(2, net, filter_size=5, in_channel=96, out_channel=256, padding='SAME')
  net = max_pool(net, ksize=3, strides=2, padding='SAME')
  net = lrn(net)

  net = conv_2d(3, net, filter_size=3, in_channel=256, out_channel=384, padding='SAME')
  net = conv_2d(4, net, filter_size=3, in_channel=384, out_channel=384, padding='SAME')
  net = conv_2d(5, net, filter_size=3, in_channel=384, out_channel=256, padding='SAME')
  net = max_pool(net, ksize=3, strides=2, padding='SAME')

  net = lrn(net)
  net = fully_connected(6, net, n_units=4096, activation='tanh')
  net = fully_connected(7, net, n_units=4096, activation='tanh')
  net = fully_connected(8, net, n_units=17, activation="", keep_prob=-1.0)
  return net