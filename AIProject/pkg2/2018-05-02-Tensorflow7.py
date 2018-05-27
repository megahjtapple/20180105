# -*- encoding: utf-8 -*-
# Youtube video: https://www.youtube.com/watch?v=jGxK7gfglrI&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=10

import tensorflow as tf

# Define a tf vaiable.
state = tf.Variable(0, name='counter')
#print(state.name)

# Define a tf constant
one = tf.constant(1)

# Define an add and update operation. Please note that this is not executed util see.run() is called.
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        #sess.run(update) Cause two things to happen:
        #1. new_value = tf.add(state, one) is executed.
        #2. update = tf.assign(state, new_value) is executed.
        #tf.add() is executed tf.assign includes new_value. new_value can be viewed as the reference of
        #tf.assign(state, new_value).
        print(sess.run(state))
        #sess.run(tf.variable) can print the value of the variable. however, if the variable is also a reference
        #of the some tf operation, if will cause the operation to run as well.























