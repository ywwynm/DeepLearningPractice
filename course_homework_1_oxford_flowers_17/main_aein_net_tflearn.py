from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.oxflower17 as oxflower17

X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))

# AeinNet
net = input_data(shape=[None, 227, 227, 3])

net = conv_2d(net, 64, 11, strides=4, activation='relu')
net = max_pool_2d(net, 3, strides=2)
net = local_response_normalization(net)

net = conv_2d(net, 256, 5, activation='relu')
net = max_pool_2d(net, 3, strides=2)
net = local_response_normalization(net)

net = fully_connected(net, 2048, activation='relu')
net = dropout(net, 0.5)
net = fully_connected(net, 17, activation='softmax')
net = regression(net, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=1e-3)

# Training
model = tflearn.DNN(net, checkpoint_path='model_aeinnet',
                    max_checkpoints=1, tensorboard_verbose=2)
model.fit(X, Y, n_epoch=20, validation_set=0.5, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='aeinnet_oxflowers17')