# https://www.youtube.com/watch?v=FTR36h-LKcY&index=14&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8

import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function=None):
    # Weight is a metric. Here tf.random_normal([in_size, out_size] create a metric which shape
    # is in_size rows and out_size columns (This is what I think. Need to verify.)
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # Biases is not a metric. It is something similar to list.
    # Biases has 1 row, out_size columns.
    # The recommended initial number for biases is none zero. This is why there is a +0.1 at the end.
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
