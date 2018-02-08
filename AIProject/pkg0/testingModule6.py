# -*- encoding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Begins.")

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

neuralNetworksLayer = 10

Weights_L1 =tf.Variable(tf.random_normal([1,neuralNetworksLayer]));

print(Weights_L1.eval())

sess.close()

print("Done.")

