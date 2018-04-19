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
epochs = 1000

train_set_1 = dataset.get_train_set(1)
train_set_1 = train_set_1.shuffle(buffer_size=10000)
train_set_1 = train_set_1.batch(batch_size)

validation_set_1 = dataset.get_validation_set(1)
validation_set_1 = validation_set_1.batch(2)

test_set_1 = dataset.get_test_set(1)
test_set_1 = test_set_1.batch(2)

X = tf.placeholder(tf.float32, [None, input_height, input_width, channel])
y_pred = an.alex_net(X)
y_true = tf.placeholder(tf.float32, [None, output_size])

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

itr_trn_1 = train_set_1.make_initializable_iterator()
next_element_trn_1 = itr_trn_1.get_next()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  start_train_time = time.time()
  print("Start to train classifier")
  for epoch in range(epochs):
    print("Epoch %d starts" % (epoch + 1))
    sess.run(itr_trn_1.initializer)
    batch_idx = 1
    while True:
      try:
        X_batch, Y_batch = sess.run(next_element_trn_1)
        real_batch_size = X_batch.shape[0]
        # print(X_batch[0][0][0][0])
        last_train_op_time = time.time()
        _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, y_true: Y_batch})
        print("epoch: %d, batch: %d, loss: %f, time cost: %f" % (epoch + 1, batch_idx, l, time.time() - last_train_op_time))
        batch_idx += 1
      except tf.errors.OutOfRangeError:  # this epoch ends
        break

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
    print("accuracy: %f" % (np.mean(pred_vals)))

  print("Classifier has been trained, total time: %f" % (time.time() - start_train_time))

  # pred_vals = []
  # while True:
  #   try:
  #     X_batch_tst, Y_batch_tst = sess.run(next_element_tst_1)
  #     pred = tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32)
  #     pred_val = sess.run(pred, feed_dict={X: X_batch_tst, y_true: Y_batch_tst})
  #     pred_vals.append(pred_val)
  #   except tf.errors.OutOfRangeError:
  #     break
  # print("accuracy: %f" % (np.mean(pred_vals)))

# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   times = []
#   cross_entropy_losses = []
#   error_rates = []
#
#   print("Start to train classifier --------------------")
#   for i in range(epoches):
#     batch_x, batch_y = mnist.train.next_batch(batch_size)
#     _, l = sess.run([optimizer, loss], feed_dict={X: batch_x, y_true: batch_y})
#     if i % 10 == 0:
#       times.append(i)
#       cross_entropy_losses.append(l)
#       pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
#       accuracy = tf.reduce_mean(tf.cast(pred, "float"))
#       accuracy_val = sess.run(accuracy, feed_dict={X: test_set_1[0], y_true: test_set_1[1]})
#       error_rates.append(1 - accuracy_val)
#       print("i: ", i, ", loss: ", l, ", error rate: ", 1 - accuracy_val)
#   print("Classifier has been trained --------------------")
#
#   plt.plot(times, cross_entropy_losses)
#   plt.xlabel('trained times')
#   plt.ylabel('cross entropy')
#   plt.show()
#
#   plt.plot(times, error_rates)
#   plt.xlabel('trained times')
#   plt.ylabel('error rate')
#   plt.show()
