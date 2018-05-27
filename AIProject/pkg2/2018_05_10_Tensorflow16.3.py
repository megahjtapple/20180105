# -*- encoding: utf-8 -*-
# Now try using the trained module to predict image.

from __future__ import print_function
import tensorflow as tf
import cv2
import numpy as np

const_trainedModuleSavePath = "trainedModule_2018_05_10_Tensorflow16.2"
const_imagePath = "images"
const_imageName = "9.png"

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
prediction = add_layer(xs, 784, 10,  activation_function=tf.nn.softmax)
predictNumber = tf.argmax(prediction, 1, name="predict")
#-----------------------------------------------------------------

# Create saver.
saver=tf.train.Saver()

# Define dir.
ckpt_dir="./" + const_trainedModuleSavePath

#--------------------------------------------------------------

# Load image.
src = cv2.imread("./" + const_imagePath + "/" + const_imageName)
cv2.imshow("Your picture", src)

# Transform to 28*28 gray image.
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.resize(src, (28, 28), interpolation=cv2.INTER_CUBIC)

# Transform to 1*784 digit data so that the module can recon it.
picture = np.zeros((28, 28))
for i in range(0, 28):
    for j in range(0, 28):
        picture[i][j] = (255 - dst[i][j])
picture = picture.reshape(1, 784)

# Metadata?
p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

#------------------------------------------------------------------------

# Load the module
sess = tf.Session()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)
saver.restore(sess,ckpt_dir+"/model.ckpt-1999") #1999 is the last step.
# Question: How does sess know that it suppose to retore the W and bias tf.variable?

# Predict:
predict_result = sess.run(predictNumber, feed_dict={xs: picture})
print("Your image is: ",predict_result[0])
cv2.waitKey(20170731)





