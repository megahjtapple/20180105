# -*- encoding: utf-8 -*-
# https://www.youtube.com/watch?v=nhn8B0pM9ls&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=16

# The core of the training is: Run tf.train.GradientDescentOptimizer(0.1).minimize(loss).
# loss is a function which contains tf.Variable. The goal is to find the best tf.Variable to minimize loss.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Make up some real data
# x_data is 3 row or 3 input sample.
# np.linespace: Return evenly spaced numbers over a specified interval.
# so np.linspace(-1,1,3) return 3 points between -1 to 1 (which is [-1, 0, 1]).
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
# In here, x_data =
# [[-1.]
#  [ 0.]
#  [ 1.]]
# y_data =
# [[ 0.5]
#  [-0.5]
#  [ 0.5]]
x_data = np.linspace(-1,1,3)[:, np.newaxis]
y_data = np.square(x_data) - 0.5

# define placeholder for inputs to network
# All data in x_data and y_data are feeded into these feeders immediately.
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layer.
# Weights_l1 has 1 row, 4 columns.
# In tf, this means there are 4 nerve cell.
Weights_l1 = tf.Variable(tf.random_normal([1, 4]))
# biases_l1 has 1 row, 4 column as well.
# When biases_l1 added into tf.matmul(xs, Weights_l1), it will be expaned to 3 rows 4 columns.
# The two new columns looks exactly the same as the original first column.
biases_l1 = tf.Variable(tf.zeros([1, 4]) + 0.1)
# tf.nn.relu is one of the activation function provided by Tensorflow.
Wx_plus_b_l1 = tf.matmul(xs, Weights_l1) + biases_l1
l1 = tf.nn.relu(Wx_plus_b_l1)

# add output layer
Weights_l2 = tf.Variable(tf.random_normal([4, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]) + 0.1)
Wx_plus_b_l2 = tf.matmul(l1, Weights_l2) + biases_l2
prediction = Wx_plus_b_l2

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
# This is the only step which causes tf.Variable to be updated.
# In this training, tf.Variable are:
# 1. W_l1, W_l2
# 2. bias_l1, bias_l2
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    # training
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 200 == 0:
        # to see the step improvement
        print("xs:\n" + str(sess.run(xs, feed_dict={xs: x_data})))
        print("ys:\n" + str(sess.run(ys, feed_dict={ys: y_data})))
        print("Weights_l1:\n" + str(sess.run(Weights_l1)))
        print("biases_l1:\n" + str(sess.run(biases_l1)))
        print("Wx_plus_b_l1:\n" + str(sess.run(Wx_plus_b_l1, feed_dict={xs: x_data})))
        print("l1:\n" + str(sess.run(l1, feed_dict={xs: x_data})))
        print("Weights_l2:\n" + str(sess.run(Weights_l2)))
        print("biases_l2:\n" + str(sess.run(biases_l2)))
        print("Wx_plus_b_l2:\n" + str(sess.run(Wx_plus_b_l2, feed_dict={xs: x_data})))
        print("prediction(l2):\n"+str(sess.run(prediction, feed_dict={xs: x_data,})))
        print("loss:\n"+str(sess.run(loss, feed_dict={xs: x_data, ys: y_data})))
        print("----------------------------------------------")
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)

raw_input("Press Enter to continue...")











