import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

num_input = 28
timestep = 28
rnn_hidden = 128
num_class = 10

input_images = tf.placeholder(tf.float32, shape=[None, timestep,num_input])
input_labels = tf.placeholder(tf.float32, shape=[None, num_class])

weight = {'out':tf.Variable(tf.random_normal([rnn_hidden,num_class]))}
bias = {'out':tf.Variable(tf.zeros([num_class]))}

def rnn_model(x, weight, bias):
    x = tf.unstack(x, timestep, axis=1)
    #print(tf.shape(x))
    lstm_cell = rnn.BasicLSTMCell(rnn_hidden, forget_bias=1.0)
    output, states = rnn.static_rnn(lstm_cell,x, dtype=tf.float32)

    return tf.matmul(output[-1], weight['out'])+bias['out']

logits = rnn_model(input_images, weight,bias)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                 labels=input_labels))

optimizor = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizor.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(input_labels,1))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1, training_steps+1):
        input_x, input_y = mnist.train.next_batch(batch_size)

        input_x = input_x.reshape((batch_size,timestep, num_input))

        sess.run(train_op, feed_dict={input_images:input_x, input_labels:input_y})

        if i==1 or i % display_step==0:
            loss, acc = sess.run([loss_op,accuracy_op], feed_dict={input_images:input_x,input_labels:input_y})
            print("Step:",i, "Minbatch Loss:", "{:.4f}".format(loss), ", Accuracy:"+"{:.3f}".format(acc))

    print("Testing")
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timestep, num_input))
    test_lable = mnist.test.labels[:test_len]
    print("Test Accuracy: ", sess.run(accuracy_op, feed_dict={input_images: test_data, input_labels: test_lable}))
