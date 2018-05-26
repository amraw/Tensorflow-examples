import tensorflow as tf
from tensorflow.contrib.factorization import KMeans
import numpy as np

# Import MNIST dataset

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

data = mnist.train.images

print("Input data shape:" + str(data.shape))

numb_steps = 50 # Number of training steps
batch_size = 1024 # batch size
k = 25 # number of clusters
number_classes = 10
numb_features = data.shape[1]


input_images = tf.placeholder(tf.float32, shape=[None, numb_features])    # Input Images to the model
input_labels = tf.placeholder(tf.float32, shape=[None, number_classes])    # Input labels to the model

# Kmeans Instance created
kmeans = KMeans(inputs=input_images, num_clusters=k, distance_metric='cosine', use_mini_batch=True)

#Build Kmeans graph
training_graph = kmeans.training_graph()

(all_scores, cluster_idx, scores, cluster_centers_initialized, cluster_centers_var, init_op, training_op) = training_graph

#print(len(cluster_idx), len(cluster_idx[0]))
cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

# Initiliaze the variables (i.e. assign there default value)
init_vars = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init_vars, feed_dict={input_images:data})
    session.run(init_op, feed_dict={input_images:data})
    for i in range(1, numb_steps+1):
        _, itr_avg_dist, image_cluster = session.run([training_op, avg_distance, cluster_idx], feed_dict={input_images:data})

        if i%10 == 0 or i == 1:
            print("Iternation %i, Average Distance %f" % (i, itr_avg_dist))
        # Count the total number of labels per centroid
    counts = np.zeros(shape=(k, number_classes))
    for i in range(len(image_cluster)):
        counts[image_cluster[i]] += mnist.train.labels[i]

    labels_map = [np.argmax(c) for c in counts]
    print(labels_map)
    labels_map = tf.convert_to_tensor(labels_map)
    # All the cluster id mapped to the class label
    cluster_lables = tf.nn.embedding_lookup(labels_map, cluster_idx)
    #Computing Accuracy
    correct_prediction = tf.equal(cluster_lables, tf.cast(tf.argmax(input_labels,1) , tf.int32))
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Test result
    test_data, test_labels = mnist.test.images, mnist.test.labels
    print("Test data accuracy: ", session.run(accuracy_op, feed_dict={input_images:test_data, input_labels:test_labels}))









