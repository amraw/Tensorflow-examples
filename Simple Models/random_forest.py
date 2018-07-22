import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=False)

num_steps = 550
batch_size = 1024
num_classes = 10
num_features = 784
num_trees = 10
max_nodes = 1000

input_image = tf.placeholder(tf.float32, shape=[None,num_features])
input_Y = tf.placeholder(tf.int32, shape=[None])

hyper_params = tensor_forest.ForestHParams(num_trees=num_trees, max_nodes=max_nodes, num_features=num_features,
                                           num_classes=num_classes).fill()
forest_graph = tensor_forest.RandomForestGraphs(hyper_params)

train_op = forest_graph.training_graph(input_image, input_Y)
loss_op = forest_graph.training_loss(input_image,input_Y)

# Measuring Accuracy

infer, _, _ = forest_graph.inference_graph(input_image)
correct_pred = tf.equal(tf.argmax(infer, 1), tf.cast(input_Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32), keep_dims=False)

init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

with tf.Session() as sess:
    sess.run(init_vars)
    for i in range(1, num_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _, loss = sess.run([train_op, loss_op], feed_dict={input_image:batch_x, input_Y:batch_y})
        if i%50==0 or i==1:
            acc = sess.run([accuracy_op], feed_dict={input_image:batch_x, input_Y:batch_y})
            print('Step: ', i, 'Loss: ', loss, 'Accuracy: ', acc)

    test_x, test_y = mnist.test.images, mnist.test.labels
    print("Test Accuracy: ", sess.run(accuracy_op, feed_dict={input_image:test_x, input_Y:test_y}))



