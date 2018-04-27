from net import conv_2d, max_pool ,lrn, fully_connected


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