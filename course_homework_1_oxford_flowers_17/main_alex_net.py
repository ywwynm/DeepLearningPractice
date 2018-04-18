import tensorflow as tf
import course_homework_1_oxford_flowers_17.dataset as dataset
import matplotlib.pyplot as plt
import course_homework_1_oxford_flowers_17.alex_net as an

learning_rate = 0.001
input_width = input_height = 224
channel = 3
output_size = 17
batch_size = 32
epochs = 1000

train_set_1 = dataset.get_train_set(1)
test_set_1 = dataset.get_test_set(1)

train_set_1 = train_set_1.shuffle(buffer_size=10000)
train_set_1 = train_set_1.repeat(epochs)
train_set_1 = train_set_1.batch(batch_size)

X = tf.placeholder(tf.float32, [None, input_height, input_width, channel])
y_pred = an.alex_net(X)
y_true = tf.placeholder(tf.float32, [None, output_size])

loss = -tf.reduce_sum(y_true * tf.nn.log_softmax(y_pred)) / batch_size
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

iterator = train_set_1.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(epochs):
    X_batch, Y_batch = sess.run(next_element)
    # print(X_batch[0][0][0][0])
    _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, y_true: Y_batch})
    if i % 10 == 0:
      # pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
      # accuracy = tf.reduce_mean(tf.cast(pred, "float"))
      # accuracy_val = sess.run(accuracy, feed_dict={X: test_set_1[0], y_true: test_set_1[1]})
      print("i: ", i, ", loss: ", l)


# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   for i in range(epoches):
#     X_batch, Y_batch = sess.run(next_element)
#     X_batch = tf.reshape(X_batch, [32, 224 * 224 * 3])
#     Y_batch = tf.one_hot(Y_batch, output_size, 1, 0)
#     _, l = sess.run([optimizer, loss], feed_dict={X: sess.run(X_batch), y_true: sess.run(Y_batch)})
#     if i % 10 == 0:
#       pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
#       # accuracy = tf.reduce_mean(tf.cast(pred, "float"))
#       # accuracy_val = sess.run(accuracy, feed_dict={X: test_set_1[0], y_true: test_set_1[1]})
#       print("i: ", i, ", loss: ", l)

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
