# -*- encoding: utf-8 -*-
# Try uing the image to recognize mnist's image.

from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#--------------------------------------------------------

const_trainedModelSavePath = "trainedModel"
const_lastTrainStepIndex = "999"

#--------------------------------------------------------
# Define nerve network.
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28

# Function for adding layer.
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

# add output layer
# softmax模型可以用来给不同的对象分配概率: https://www.cnblogs.com/flyu6/p/5555178.html
# Therefore, this prediction is a probability.
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
predictNumber = tf.argmax(prediction, 1, name="predict")
#-----------------------------------------------------------------

# Create saver.
saver=tf.train.Saver()

# Define dir.
ckpt_dir="./" + const_trainedModelSavePath

#--------------------------------------------------------------

# Load the model
sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess,ckpt_dir+"/model.ckpt-" + const_lastTrainStepIndex)
# Question: How does sess know that it suppose to restore the W and bias tf.variable?

#-----------------------------------------------------------------------

# Load image from mnist.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(1)
batchXsPic = np.zeros((28, 28))
pIndex = 0
for i in range(0, 28):
    for j in range(0, 28):
        batchXsPic[i][j] = (batch_xs[0][pIndex])
        pIndex = pIndex + 1
cv2.imshow("batchXsPic", batchXsPic)
print("The image read from mnist is:" + str(sess.run(tf.argmax(batch_ys, 1, name="targetedNum"))))

#-----------------------------------------------------------------------


# Predict:
predict_result = sess.run(predictNumber, feed_dict={xs: batch_xs})
print("Use model to predict image. Result: ",predict_result[0])
cv2.waitKey(3000)








