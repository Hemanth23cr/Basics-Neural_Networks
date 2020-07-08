"""
Hemanth CR | hemanth23cr@gmail.com
Addition Of Two Numbers: 15 and 25 (constants)
Expctd Result: 40
"""

# Importing the Tensorflow Library

import tensorflow as tf

# An instance of a session is created using tf.Session() call.

with tf.compat.v1.Session() as sess:

# Assuming the numbers as tensorflow constants
    a = tf.constant(15)
    b = tf.constant(25)
    c = a + b


# Now we run the session and store the results of the computation in 'output'
    output = sess.run(c)
    print("Value of C after running the session: ",output)

# Closing the session
sess.close()

