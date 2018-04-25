import tensorflow as tf
import numpy as np
import course_homework_1_oxford_flowers_17.dataset as dataset
import matplotlib.pyplot as plt
import course_homework_1_oxford_flowers_17.alex_net as an

import time

train_set_size = 680
test_set_size = 340

learning_rate = 1e-3
input_width = input_height = 224
channel = 3
output_size = 17
batch_size = 64
epochs = 1

train_set_1 = dataset.get_train_set(1)
train_set_1 = train_set_1.shuffle(buffer_size=10000)
train_set_1 = train_set_1.batch(batch_size)

validation_set_1 = dataset.get_validation_set(1)
validation_set_1 = validation_set_1.batch(340)

test_set_1 = dataset.get_test_set(1)
test_set_1 = test_set_1.batch(2)

X = tf.placeholder(tf.float32, [None, input_height, input_width, channel])
y_pred = an.alex_net(X)
y_true = tf.placeholder(tf.float32, [None, output_size])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)

itr_trn_1 = train_set_1.make_initializable_iterator()
next_element_trn_1 = itr_trn_1.get_next()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  start_train_time = time.time()
  accuracies = []
  losses = []
  epochs_arr = []
  epochs_10_arr = []
  print("Start to train classifier")
  for epoch in range(epochs):
    print("------------------ Epoch %d starts ------------------" % (epoch + 1))
    sess.run(itr_trn_1.initializer)
    batch_idx = 1
    while True:
      try:
        X_batch, Y_batch = sess.run(next_element_trn_1)
        # print(X_batch[0][0][0][0])
        last_train_op_time = time.time()
        _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, y_true: Y_batch})
        losses.append(l)
        epochs_arr.append(epoch)
        print("epoch: %d, batch: %d, loss: %f, time cost: %f" % (epoch + 1, batch_idx, l, time.time() - last_train_op_time))
        batch_idx += 1
      except tf.errors.OutOfRangeError:  # this epoch ends
        break

    if (epoch + 1) % 10 == 0:
      print()
      print("------------------ evaluating accuracy on validation set ------------------")
      itr_val_1 = validation_set_1.make_one_shot_iterator()
      next_element_val_1 = itr_val_1.get_next()
      pred_vals = []
      while True:
        try:
          X_batch_tst, Y_batch_tst = sess.run(next_element_val_1)
          pred = tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32)
          pred_val = sess.run(pred, feed_dict={X: X_batch_tst, y_true: Y_batch_tst})
          pred_vals.append(pred_val)
        except tf.errors.OutOfRangeError:
          break
      accuracy = np.mean(pred_vals)
      print("accuracy: %f" % accuracy)
      print()
      accuracies.append(accuracy)
      epochs_10_arr.append(epoch + 1)

  print("Classifier has been trained, total time: %f" % (time.time() - start_train_time))

  import os
  if not os.path.exists("result"):
    os.mkdir("result")

  post_fix = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

  plt.plot(epochs_arr, losses)
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.savefig("result/alex_net_loss_" + post_fix + ".png")

  plt.plot(epochs_10_arr, accuracies)
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.savefig("result/alex_net_acc_" + post_fix + ".png")

