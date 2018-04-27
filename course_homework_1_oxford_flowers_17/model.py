import tensorflow as tf
import dataset
import aein_net, alex_net, vgg_16
import utils
import time

def train_and_evaluate(net_name="aein_net", epochs = 250, train_batch_size=64, learning_rate=1e-3, optimizer="adam"):
  train_set_size = 680
  val_test_set_size = 340

  input_width = input_height = 224
  channel = 3
  output_size = 17

  train_set_1 = dataset.get_train_set(1)
  train_set_1 = train_set_1.shuffle(buffer_size=10000)
  train_set_1 = train_set_1.batch(train_batch_size)

  validation_set_1 = dataset.get_validation_set(1)
  validation_set_1 = validation_set_1.batch(68)

  test_set_1 = dataset.get_test_set(1)
  test_set_1 = test_set_1.batch(68)

  X = tf.placeholder(tf.float32, [None, input_height, input_width, channel])
  if net_name == 'aein_net':
    y_pred = aein_net.aein_net(X)
  elif net_name == 'alex_net':
    y_pred = alex_net.alex_net(X)
  elif net_name == 'vgg_16':
    y_pred = vgg_16.vgg_16(X)
  else:
    y_pred = aein_net.aein_net(X)
  y_true = tf.placeholder(tf.float32, [None, output_size])

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))

  if optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif optimizer == 'momentum':
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
  elif optimizer == 'gradient_descent':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    optimizer = tf.train.AdamOptimizer(learning_rate)
  optimizer = optimizer.minimize(loss)

  itr_trn_1 = train_set_1.make_initializable_iterator()
  next_element_trn_1 = itr_trn_1.get_next()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start_train_time = time.time()
    accuracies = []
    losses = []
    epochs_arr = []
    epochs_10_arr = []
    print("Start to train classifier, model: " + net_name)
    for epoch in range(epochs):
      print("------------------ Epoch %d starts ------------------" % (epoch + 1))
      sess.run(itr_trn_1.initializer)
      batch_idx = 1
      while True:
        try:
          X_batch, Y_batch = sess.run(next_element_trn_1)
          last_train_op_time = time.time()
          _, l = sess.run([optimizer, loss], feed_dict={X: X_batch, y_true: Y_batch})
          losses.append(l)
          epochs_arr.append(epoch)
          print("epoch: %d, batch: %d, loss: %f, time cost: %f" % (
          epoch + 1, batch_idx, l, time.time() - last_train_op_time))
          batch_idx += 1
        except tf.errors.OutOfRangeError:  # this epoch ends
          break

      if (epoch + 1) % 10 == 0:
        print()
        print("------------------ evaluating accuracy on validation set ------------------")
        accuracy = utils.evaluate(validation_set_1, sess, X, y_true, y_pred)
        print("accuracy: %f" % accuracy)
        print()
        accuracies.append(accuracy)
        epochs_10_arr.append(epoch + 1)

    print("Classifier has been trained, total time: %f" % (time.time() - start_train_time))
    print()
    print("------------------ evaluating accuracy on test set ------------------")
    print("accuracy: %f" % utils.evaluate(test_set_1, sess, X, y_true, y_pred))
    print()

    utils.save_result(net_name, epochs_arr, losses, epochs_10_arr, accuracies)