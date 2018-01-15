# -*- encoding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("Begins.")

checkpoint_dir = "./"

#200 random range from -0.5 to 0.5 in one column.
#x_data is uniform distributed. Values are always the same.
x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
print("x_data:")
print(str(x_data))
#noise are some random numbers. Values are different each time.
noise = np.random.normal(0, 0.02, x_data.shape)
#y_data = x_data^2 + noise
y_data = np.square(x_data) + noise
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
# tf.matmul(x, Weights_L1) 是指 w 乘以 x。不过根据网上找到的资料，很多例子用的都是 xw。
Weights_L1 =tf.Variable(tf.random_normal([1,10]));
biases_L1 =tf.Variable(tf.zeros([1,10]));
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 下面这个是算切线的？为什么要这么做？
L1 = tf.nn.tanh(Wx_plus_b_L1)

#Output layer
Weights_L2 =tf.Variable(tf.random_normal([10,1]));
biases_L2 =tf.Variable(tf.zeros([1,1]));
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#loss function. What's this for?
loss = tf.reduce_mean(tf.square(y-prediction))

#Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

with tf.Session() as sess:

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print("发现之前训练好的模型。")
        saver.restore(sess, ckpt.model_checkpoint_path)

        x_test = tf.float32([0.25, 1])

    else:
        print("进行训练。")

        #Init variables
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            sess.run(train_step, feed_dict={x:x_data, y:y_data})

        #Get predicted graph
        prediction_value = sess.run(prediction, feed_dict={x:x_data})

        #Draw graph
        plt.figure();
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        #plt.show Doesn't work in Eclipse.
        plt.savefig("graph0.svg")

        saver.save(sess, checkpoint_dir + "Model.ckpt")

print("Done.")
























