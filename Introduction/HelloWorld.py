# Hemanth CR| hemanth23cr@gmail.com


import tensorflow as tf

# TensorFlow is an open-source machine learning library for research and production

with tf.compat.v1.Session() as sess:

# tf.constant() holds constant values

    hello = tf.constant('Hello, TensorFlow!')

# Starting a session in which the computations take place
# The code creates a session instance, sess, using tf.Session()

# The sess.run() function then evaluates the tensor and returns the results, wich are then printed
    print(sess.run(hello))


sess.close()
