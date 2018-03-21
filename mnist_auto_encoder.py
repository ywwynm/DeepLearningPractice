import tensorflow as tf

text = tf.constant("Hello Deep Learning")
with tf.Session() as sess:
    print(sess.run(text))