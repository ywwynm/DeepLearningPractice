import os, time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def save_result(epochs_arr, losses, epochs_10_arr, accuracies):
  if not os.path.exists("result"):
    os.mkdir("result")

  post_fix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

  plt.plot(epochs_arr, losses)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.savefig("result/alex_net_loss_" + post_fix + ".png")

  plt.clf()  # clear existing figure content
  plt.plot(epochs_10_arr, accuracies)
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.savefig("result/alex_net_acc_" + post_fix + ".png")


def evaluate(dataset, sess, X, y_true, y_pred):
  itr = dataset.make_one_shot_iterator()
  next_element = itr.get_next()
  pred_vals = []
  while True:
    try:
      X_batch_val, Y_batch_val = sess.run(next_element)
      pred = tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32)
      pred_val = sess.run(pred, feed_dict={X: X_batch_val, y_true: Y_batch_val})
      pred_vals.append(pred_val)
    except tf.errors.OutOfRangeError:
      break
  return np.mean(pred_vals)