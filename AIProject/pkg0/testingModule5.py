# -*- encoding: utf-8 -*-

# 研究怎么打印 Tensorflow 变量。
# 参考：https://vimsky.com/article/3675.html

# 无论何时评估计算图(使用run或eval)，查看张量值的最简单方法是使用Print操作，如下例所示：

# Initialize session
import tensorflow as tf
sess = tf.InteractiveSession()

# Some tensor we want to print the value of
a = tf.constant([1.0, 3.0])

# Add print operation
a = tf.Print(a, [a], message="This is a: ")

# Add more elements of the graph using a
b = tf.add(a, a).eval()

# 现在，每当我们评估整个计算图时，例如使用b.eval()，我们得到：

# I tensorflow/core/kernels/logging_ops.cc:79] This is a: [1 3]

#----------------------------------------------------------------

# 再次强调，不可能在没有运行图的情况下检查值。

# 任何寻找打印值的简单示例的代码如下所示。代码可以在ipython notebook中不做任何修改的情况下执行

import tensorflow as tf

#define a variable to hold normal random values
normal_rv = tf.Variable( tf.truncated_normal([2,3],stddev = 0.1))

#initialize the variable
init_op = tf.initialize_all_variables()

#run the graph
with tf.Session() as sess:
    sess.run(init_op) #execute init_op
    #print the random values that we sample
    print (sess.run(normal_rv))

# 输出：

# [[-0.16702934  0.07173464 -0.04512421]
#  [-0.02265321  0.06509651 -0.01419079]]

