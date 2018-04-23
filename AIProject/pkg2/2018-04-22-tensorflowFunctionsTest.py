# -*- encoding: utf-8 -*-
# 2018-04-22

from __future__ import print_function
import tensorflow as tf
import numpy as np

# Print 10 random numbers.
for step in range(10):

    # https://www.tensorflow.org/api_docs/python/tf/random_uniform
    # https://crypto.stackexchange.com/questions/20839/what-is-the-difference-between-uniformly-and-at-random-in-crypto-definitions
    #
    # random_uniform
    #
    # random 指从 Sample Set 里随机抽取一定数量的 Samples。
    #
    # uniform 指 Sample 是均匀分布的。
    # Samples 在 Set 里面是均匀分布的。因此每个 Set 里面的 Sample 被取得的几率都是相等的。
    # 比如，Set 里有 4 个 Samples。如果 Samples 是均匀分布的，则每个 Sample 被取得的几率都是 1/4。
    #
    # 所以，random_uniform 最终就相当于在规定范围内随机取数。


    Weights = tf.Variable(tf.random_uniform([5], -1.0, 1.0))

    # Init sess. This have to be done before using Weight.
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print (sess.run(Weights));
















