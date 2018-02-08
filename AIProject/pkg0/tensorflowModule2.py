# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Begins.")

neuralNetworksLayer = 10

# 构建输入数据和预期输出数据。
x_data = np.linspace(-0.2, 0.2, 5)[:,np.newaxis]
print("x_data:")
print(str(x_data))
y_data = np.square(x_data)
print("y_data:")
print(str(y_data))

# 这里的 None 应该是指任何形状。可以理解成矩阵里面的一个任意长度的行。
# None 这个位置应该就是用来插入 x_data 的。
# 因此，最终的结果应该是做出 x_data 行 1 列的数据。
# 这和上面 x_data 的形状对上了。
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])


#Mid layer
#- tf.random_normal([1,10]) 里面的 1 代表着输入层只有一个神经元
# [1,10]指的是 1 行 10 列。
#  10 代表中间层有 10 个神经元。
# tf.matmul(x, Weights_L1) 是指 x 乘以 w。
Weights_L1 =tf.Variable(tf.random_normal([1,neuralNetworksLayer]));
biases_L1 =tf.Variable(tf.zeros([1,neuralNetworksLayer]));
# 问题：如果 Weights_L1 是 neuralNetworksLayer x neuralNetworksLayer，
# 为什么 biases_L1 是 1 x neuralNetworksLayer？
# 他们为什么可以相加？
# 很可能 x 不是 [n, 1] 而是 [1, 1]。也就是说，x_data 到 x 是一个一个地进去的。
# 根据打印出来的 Weights_L1 和 biases_L1 的形状来看，x 应该就是 [1, 1]。
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# tf.nn.tanh 是激活函数。
L1 = tf.nn.tanh(Wx_plus_b_L1)

#Output layer
Weights_L2 =tf.Variable(tf.random_normal([neuralNetworksLayer,1])); #[10,1] 10 行 1 列。
biases_L2 =tf.Variable(tf.zeros([1,1])); #输出层只有一个神经元。
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#loss function. What's this for?
loss = tf.reduce_mean(tf.square(y-prediction))

#Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

print("进行训练。")
with tf.Session() as sess:

    #Init variables
    sess.run(tf.global_variables_initializer())

    #训练次数
    for _ in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
        print("Data in training:")
        print("Weights_L1:")
        print(sess.run(Weights_L1))
        print("biases_L1:")
        print(sess.run(biases_L1))
        print("-------------------------------------")

    #Get predicted graph
    prediction_value = sess.run(prediction, feed_dict={x:x_data})

    #Draw graph
    plt.figure();
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    #plt.show Doesn't work in Eclipse.
    plt.savefig("graph0.svg")

print("Done.")
























