import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = './mlp_model.ckpt'

num_hidden_1 = 256
num_hidden_2 = 256
num_input = 784
num_classes = 10

input_images = tf.placeholder(tf.float32, shape=[None, num_input])
input_label = tf.placeholder(tf.float32, shape=[None,num_classes])

def multi_layer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])

    return out_layer


weights = {'w1': tf.Variable(tf.random_normal([num_input,num_hidden_1])),
           'w2': tf.Variable(tf.random_normal([num_hidden_1,num_hidden_2])),
           'w3': tf.Variable(tf.random_normal([num_hidden_2, num_classes]))
           }

biases = {'b1': tf.Variable(tf.random_normal([num_hidden_1])),
          'b2': tf.Variable(tf.random_normal([num_hidden_2])),
          'b3': tf.Variable(tf.random_normal([num_classes]))
         }

logits = multi_layer_perceptron(input_images, weights, biases)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_label))

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(3):
        avg_loss = 0
        batch_number = int(mnist.train.num_examples/batch_size)
        for i in range(batch_number):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            ls, _ = sess.run([loss, training_op], feed_dict={input_images: batch_x, input_label: batch_y})

            avg_loss += ls/batch_size

        if epoch % display_step == 0:
            print('Epoch:', epoch, "Loss:", "{:.9f}".format(avg_loss))

    print("Finished Optimization")
    correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(input_label,1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    print("Test Accuracy: ", sess.run(accuracy_op, feed_dict={input_images:mnist.test.images,
                                                              input_label:mnist.test.labels}))
    save_path = saver.save(sess,model_path)

    print("Model Saved to file: %s" % save_path)

print("Starting Session 2")

with tf.Session() as sess:
    sess.run(init)

    saver.restore(sess, model_path)
    for epoch in range(7):
        avg_loss = 0
        batch_number = int(mnist.train.num_examples/batch_size)
        for i in range(batch_number):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            ls, _ = sess.run([loss, training_op], feed_dict={input_images: batch_x, input_label: batch_y})

            avg_loss += ls/batch_size

        if epoch % display_step == 0:
            print('Epoch:', epoch, "Loss:", "{:.9f}".format(avg_loss))
    print("Finished Second Optimization")
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(input_label, 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print("Test Accuracy: ", sess.run(accuracy_op, feed_dict={input_images: mnist.test.images,
                                                              input_label: mnist.test.labels}))