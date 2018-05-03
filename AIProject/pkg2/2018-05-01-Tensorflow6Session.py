# -*- encoding: utf-8 -*-
# Youtube video: https://www.youtube.com/watch?v=HhjtJ73AwIY&list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8&index=9

import tensorflow as tf

# 1 row 2 columns metric.
matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2) #matrix multiply np.dor(m1, m2)

#method1
#sess = tf.Session()
#result = sess.run(product)
#print(result)
#sess.close()

#method2
with tf.Session() as sess:
    result = sess.run(product)
    print(result)






























