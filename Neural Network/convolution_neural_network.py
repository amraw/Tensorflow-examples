import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=False)

learning_rate = 0.001
numb_steps = 1500
batch_size = 256

numb_features = 784
numb_class = 10
dropout = 0.5


def create_conv_network(input, numb_class, dropout, reuse, is_training):
    with tf.variable_scope('ConvNet', reuse=reuse):
        images = input['images']
        images = tf.reshape(images, shape=[-1, 28, 28, 1])
        conv_layer_1 = tf.layers.conv2d(images, filters=32, kernel_size=5, activation=tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling2d(conv_layer_1, 2, 2)

        conv_layer_2 = tf.layers.conv2d(max_pool_1, 64, 3, activation=tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling2d(conv_layer_2, 2, 2)

        flatten_layer_1 = tf.contrib.layers.flatten(max_pool_2)
        flatten_layer_1 = tf.layers.dense(flatten_layer_1, 1024)
        flatten_layer_1 = tf.layers.dropout(flatten_layer_1, rate=dropout, training=is_training)

        output = tf.layers.dense(flatten_layer_1, numb_class)

    return output


def cnn_model(features,labels,mode):

    logits_train = create_conv_network(features, numb_class, dropout, False, True)

    logits_test = create_conv_network(features, numb_class, dropout, True, False)

    pred_class = tf.argmax(logits_test, axis=1)
    pred_prob = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_class)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train,
                                                                     labels=tf.cast(labels, dtype=tf.int32)))
    optimizor = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizor.minimize(loss_op, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_class)

    train_est = tf.estimator.EstimatorSpec(mode=mode,loss=loss_op, train_op=train_op, predictions=pred_class,
                                           eval_metric_ops={'accuracy':acc_op})

    return train_est


model = tf.estimator.Estimator(cnn_model)

train_input = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images}, y=mnist.train.labels,
                                                 batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(train_input, steps=numb_steps)

test_input = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images}, y=mnist.test.labels,
                                                batch_size=batch_size, shuffle=False)

test_out = model.evaluate(test_input)

print("Test Accuracy: ", test_out['accuracy'])

