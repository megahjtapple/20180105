# -*- encoding: utf-8 -*-
# https://www.youtube.com/watch?v=FTR36h-LKcY&index=14&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# Make up some real data
# x_data is 300 row or 300 input sample.
# np.linespace: Return evenly spaced numbers over a specified interval.
# so np.linspace(-1,1,300) return 300 points between -1 to 1.
# [:,np.newaxis] turns an array into [n x 1] metric where n is number of rows.
# Example:
# a=np.array([1,2,3,4,5])
# b=a[:,np.newaxis]
# print a.shape,b.shape
# print a
# print b
# Output:
# (5,) (5, 1)
# [1 2 3 4 5]
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]]
x_data = np.linspace(-1,1,300)[:, np.newaxis]
# x_data.shape means the same shape as x_data.
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer
# Input is x_data which has only one data (each time?)
# Output is 10 which means that the hidden layer has 10 nerve cells.
# tf.nn.relu is one of the activation function provided by Tensorflow.
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)

# add output layer
# Hmmm... why output layer doesn't need any activation functions?
# Input data is 10 this time since the hidden has 10 nerve cells which means that
# it has 10 outputs.
prediction = add_layer(l1, 10, 1, activation_function=None)

# the error between prediction and real data
# reduction_indices is the old name for axis.
# reduction_indices d0 - rows. d1 - columns.
#   ┌       ┐
# d0│[1 1 1]│
#   │[1 1 1]│
#   └  d1   ┘
# reduction_indices=[1] erase d1 which makes it:
#   ┌   ┐
# d0│[3]│
#   │[3]│
#   └   ┘
# But tf make shows [3, 3] instead when printing.
# I guess reduction_indices=[1] means erase.
# Why calcualting mean here?
# There are too many calculation here. I need to think about whether I should investigate it.
# 1. The best case is I understand what's happening.
# 2. If not, try looking at another examples to see whether this is the standard way to do it.
# If so, may be I can just treat it as a black box and only focus on the input and output.
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# 0.1 is learning efficiency. < 1 usually.
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))













