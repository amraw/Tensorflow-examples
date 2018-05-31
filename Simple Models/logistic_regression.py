import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# Load the dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.01
training_epochs = 30
batch_size = 128
error_display_step = 1

number_features = mnist.train.images.shape[1]
number_classes = mnist.train.labels.shape[1]

input_X = tf.placeholder(tf.float32, shape=[None, number_features])
input_Y = tf.placeholder(tf.float32, shape=[None, number_classes])

W = tf.Variable(tf.zeros([number_features, number_classes]))
b = tf.Variable(tf.zeros([number_classes]))

pred = tf.nn.softmax(tf.add(tf.matmul(input_X, W), b))

cost = tf.reduce_mean(-tf.reduce_sum(input_Y * tf.log(pred), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for epoch in range(1, training_epochs+1):
        numb_batches = int(len(mnist.train.images)/batch_size)
        avg_cost = 0
        for index in range(numb_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)

            _,c = session.run([optimizer, cost], feed_dict={input_X:batch_x, input_Y: batch_y})
            avg_cost+= (c / batch_size)
        if epoch % error_display_step == 0:
            print("Epoch: %04d" % (epoch), "cost=","{:.9f}".format(avg_cost))


    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(input_Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy on the Test data: %0.9f" % accuracy.eval({input_X:mnist.test.images, input_Y: mnist.test.labels}))




