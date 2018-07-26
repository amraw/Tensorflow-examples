import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot=False)

learning_rate = 0.01
numb_steps = 1000
batch_size = 128
display_step = 100

hidden1_dim = 256
hideen2_dim = 256

num_input = 784
num_classes = 10

def neural_network(input):
    x = input['images']

    layer_1 = tf.layers.dense(x,hidden1_dim)

    layer_2 = tf.layers.dense(layer_1, hideen2_dim)

    output = tf.layers.dense(layer_2,num_classes)

    return output

def feed_forward_model(features, labels,mode):

    logits = neural_network(features)

    pred_class = tf.argmax(logits, axis=1)
    pred_probs = tf.nn.softmax(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_class)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int32),
                                                                         logits=logits))

    optimizor = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizor.minimize(loss, global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_class)

    estimate_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_class, loss=loss, train_op=train_op,
                                               eval_metric_ops={'accuracy':acc_op})

    return estimate_spec


model = tf.estimator.Estimator(feed_forward_model)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.train.images}, y=mnist.train.labels, batch_size=batch_size,
                                              num_epochs=None, shuffle=True)

model.train(input_fn, steps=numb_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(x={'images':mnist.test.images}, y=mnist.test.labels, batch_size=batch_size,
                                              shuffle=False)

e = model.evaluate(input_fn)

print("Test Data Accuracy: ", e['accuracy'])


