# -*- encoding: utf-8 -*-

import tensorflow as tf

# 1 row 2 columns metric.
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2) #matrix multiply np.dor(m1, m2)

#method1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()


