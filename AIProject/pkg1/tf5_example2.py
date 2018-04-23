# -*- encoding: utf-8 -*-

# View more python tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)

# 这里的 0.1 和 0.3 就是训练的目标了。
# 0.1 是 Weight，0.3 是 biases。
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
# https://www.tensorflow.org/api_docs/python/tf/random_uniform
#
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
### create tensorflow structure end ###

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))


















