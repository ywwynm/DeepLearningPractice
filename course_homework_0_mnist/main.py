import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import course_homework_0_mnist.auto_encoder as ae

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

encoded_data = ae.get_encoded_data(mnist, mnist.test.images)
