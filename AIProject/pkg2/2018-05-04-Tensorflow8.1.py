# -*- encoding: utf-8 -*-
#Was trying to print fed dict but failed.

import tensorflow as tf
import numpy as np

data_feeding = np.linspace(-2,2,5)[:, np.newaxis]
input1 = tf.placeholder(tf.float32, [None, 1])

# Trying to print the value of the place holder.
input1Variable = tf.Variable([0.0], name='input1Variable')[:, np.newaxis]
copyInput1ToVariable = tf.assign(input1Variable, input1)

with tf.Session() as sess:
    # Print value assigned into input1 and 2:
    print("Value fed into input1:" + str(sess.run(copyInput1ToVariable, feed_dict={input1:data_feeding})))