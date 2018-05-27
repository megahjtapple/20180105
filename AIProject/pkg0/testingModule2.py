# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 研究：tensorflow 在做矩阵相乘的时候，
# 到底那些矩阵的样子是怎样的？

print("Begins.")

a = np.linspace(0, 5, 6)[:,np.newaxis]

print("a:\n" + str(a))

b = np.linspace(0, 5, 6)[np.newaxis,:]

print("b:\n" + str(b))

matrixProductTFStep = tf.matmul(a, b)

sess = tf.Session()
matrixProduct = sess.run(matrixProductTFStep)
print(str(matrixProduct))

#with tf.Session() as sess:
#    matrixProduct = sess.run(matrixProductTFStep)
#    print(str(matrixProduct))














