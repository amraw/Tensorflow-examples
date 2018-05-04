import tensorflow as tf

mat1 = tf.constant([[2., 2.]])
mat2 = tf.constant([[2.], [2.]])

product = tf.matmul(mat1, mat2)

with tf.Session() as sess:
    print("Product of matrix: %i " % sess.run(product))