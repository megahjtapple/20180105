# -*- encoding: utf-8 -*-
# https://www.youtube.com/watch?annotation_id=annotation_279191505&feature=iv&index=18&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&src_vid=jGxK7gfglrI&v=fCWbRboJ4Rs

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
# I was trying to print the value of the place holder but it doesn't work out.
#input1Variable = tf.Variable(0.0, name='input1Variable')
#input2Variable = tf.Variable(0.0, name='input2Variable')
#copyInput1ToVariable = tf.assign(input1Variable, input1)
#copyInput2ToVariable = tf.assign(input2Variable, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
    # This feed_dict cause "shape not match" error?
    #sess.run(copyInput1ToVariable, feed_dict={input1: [7.]})
    #sess.run(copyInput2ToVariable, feed_dict={input2: [2.]})