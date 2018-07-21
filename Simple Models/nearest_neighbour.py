import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)


x_train, y_train = mnist.train.next_batch(5000)
x_test, y_test = mnist.test.next_batch(200)

train_input = tf.placeholder(tf.float32, shape=[None, 784])
test_input = tf.placeholder(tf.float32, shape=[784])

distance = tf.reduce_sum(tf.abs(tf.add(train_input, tf.negative(test_input))), reduction_indices=1)

pred = tf.arg_min(distance, 0)

accuracy = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(len(x_test)):
        nearest_index = sess.run(pred, feed_dict={train_input:x_train, test_input:x_test[i, :]})
        print("Test", i, "Predicted:", np.argmax(y_train[nearest_index]), "Actual:", np.argmax(y_test[i]))
        if np.argmax(y_train[nearest_index]) == np.argmax(y_test[i]):
            accuracy += 1./len(x_test)

print("Accuracy:", accuracy)