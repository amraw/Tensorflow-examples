import tensorflow as tf
import numpy as np

# Model parameters
learning_rate = 0.01
number_epoch = 1000
display_step = 50

train_data_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997,
                           5.654, 9.27, 3.1])
train_data_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904,
                           2.42, 2.94, 1.3])

numb_samples = train_data_X.shape[0]

input_X = tf.placeholder(dtype=tf.float32)
input_Y = tf.placeholder(dtype=tf.float32)

# Weigth Matrix and bias
W = tf.Variable(initial_value=np.random.rand(), name="weights")   # By default trainable is set true
b = tf.Variable(initial_value=np.random.rand(), name="bias")      # By default trainable is set true

# Calculating Y

predicted_Y = tf.add(tf.multiply(input_X, W), b)

# Error calculated is Mean Square Error
cost = tf.reduce_sum(tf.pow(predicted_Y-input_Y, 2) / (2 * numb_samples))

# Gradient descent is used for optimization

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for epoch in range(number_epoch):
        for x, y in zip(train_data_X, train_data_Y):
            session.run(optimizer, feed_dict={input_X:x, input_Y:y})
            if (epoch+1) % display_step == 0:
                cost_per_epoch = session.run(cost, feed_dict={input_X:train_data_X, input_Y:train_data_Y})
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_per_epoch),"W=", session.run(W),
                      "b=", session.run(b))

    training_cost = session.run(cost, feed_dict={input_X:train_data_X, input_Y:train_data_Y})
    print("\nTraining cost=", training_cost, "W=", session.run(W), "b=", session.run(b))

    # Testing data
    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    # Calculating Mean Square error

    test_cost = tf.reduce_sum(tf.pow(predicted_Y - input_Y, 2) /(2 * test_X.shape[0]))

    test_cost_cal_val = session.run(test_cost, feed_dict={input_X:test_X, input_Y:test_Y})

    print("\nTesting cost=", test_cost_cal_val)

    print("\nAbsolute mean square loss difference:", abs(training_cost - test_cost_cal_val))



