import tensorflow as tf

const = tf.constant("hello world")

with tf.Session() as tf_session:
    print(tf_session.run(const))



