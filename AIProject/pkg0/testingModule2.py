# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Begins.")

a = np.linspace(0, 5, 6)[:,np.newaxis]

print("a:\n" + str(a))

b = np.linspace(0, 5, 6)[np.newaxis,:]

print("b:\n" + str(b))

matrixProductTFStep = tf.matmul(a, b)

with tf.Session() as sess:
    matrixProduct = sess.run(matrixProductTFStep)
    print(str(matrixProduct))














