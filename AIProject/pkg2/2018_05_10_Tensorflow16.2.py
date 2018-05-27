# -*- encoding: utf-8 -*-
# https://www.youtube.com/watch?v=aNjdw9w_Qyc&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=21

from __future__ import print_function
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

const_trainedModuleSavePath = "trainedModule_2018_05_10_Tensorflow16.2"

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    # y_pre is probability.
    # There are 10 elem in a row, but they are not 0 or 1. They are probabilities. The highest probability
    # indicates that this one is very likely to be one. For example, [x x 0.99 x ...] means #3 is very
    # likely to be 1 so predition is 3.
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # Output the index of the highest number in the metrix.
    # Example:
    # A = [[1,3,4,5,6]]
    # B = [[1,3,4], [2,4,1]]
    #
    # with tf.Session() as sess:
    #     print(sess.run(tf.argmax(A, 1)))
    #     print(sess.run(tf.argmax(B, 1)))
    # output:
    # [4] (6 index is 4)
    # [2 1] (4 index is 2; 4 index is 1)
    # Reference:https://blog.csdn.net/UESTC_C2_403/article/details/72232807
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
# softmax模型可以用来给不同的对象分配概率: https://www.cnblogs.com/flyu6/p/5555178.html
# Therefore, this prediction is a probability.
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)

# the error between prediction and real data
# What's this? How does this work? This seems to be important but mofan didn't explain in the video.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Create saving dir
ckpt_dir="./" + const_trainedModuleSavePath
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

# Create saver
saver=tf.train.Saver()

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(20000):
    # batch_ys:
    # [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    #  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
    #  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
    # This is basically using index to mark the number.
    # for example:
    # [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.] => 2
    # [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.] => 3
    # Not sure batch_xs. But I think it is the 784 digits picture.
    # using 0 and 1 to represent.
    batch_xs, batch_ys = mnist.train.next_batch(10000)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    # Save trainging
    saver.save(sess,ckpt_dir+"/model.ckpt",global_step=i)
    if i % 200 == 0:
        print("compute_accuracy:" + str(compute_accuracy(
            mnist.test.images, mnist.test.labels)))

print("compute_accuracy(Final):" + str(compute_accuracy(
            mnist.test.images, mnist.test.labels)))






