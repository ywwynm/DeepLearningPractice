from net import conv_2d, max_pool, fully_connected

def vgg_16(input):
  net = conv_2d(1, input, filter_size=3, in_channel=3, out_channel=64, strides=1)
  net = conv_2d(2, net, filter_size=3, in_channel=64, out_channel=64, strides=1)
  net = max_pool(net, ksize=2, strides=2)

  net = conv_2d(3, net, filter_size=3, in_channel=64, out_channel=128, strides=1)
  net = conv_2d(4, net, filter_size=3, in_channel=128, out_channel=128, strides=1)
  net = max_pool(net, ksize=2, strides=2)

  net = conv_2d(5, net, filter_size=3, in_channel=128, out_channel=256, strides=1)
  net = conv_2d(6, net, filter_size=3, in_channel=256, out_channel=256, strides=1)
  net = conv_2d(7, net, filter_size=3, in_channel=256, out_channel=256, strides=1)
  net = max_pool(net, ksize=2, strides=2)

  net = conv_2d(8, net, filter_size=3, in_channel=256, out_channel=512, strides=1)
  net = conv_2d(9, net, filter_size=3, in_channel=512, out_channel=512, strides=1)
  net = conv_2d(10, net, filter_size=3, in_channel=512, out_channel=512, strides=1)
  net = max_pool(net, ksize=2, strides=2)

  net = conv_2d(11, net, filter_size=3, in_channel=512, out_channel=512, strides=1)
  net = conv_2d(12, net, filter_size=3, in_channel=512, out_channel=512, strides=1)
  net = conv_2d(13, net, filter_size=3, in_channel=512, out_channel=512, strides=1)
  net = max_pool(net, ksize=2, strides=2)

  net = fully_connected(14, net, n_units=4096, activation='relu')
  net = fully_connected(15, net, n_units=4096, activation='relu')
  net = fully_connected(16, net, n_units=17, activation="", keep_prob=-1.0)

  return net