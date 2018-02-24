import tensorflow as tf
import numpy as np

matrix1 = np.array([[1,2,3], [4,5,6]])
matrix2 = np.array([[1,2], [3,4], [5,6]])


a = tf.constant(6.5, name="constant_a")
b = tf.constant(3.4, name="constant_b")
c = tf.constant(3.0, name="constant_c")
d = tf.constant(100.2, name="constant_d")

mult = tf.matmul(matrix1, matrix2, name="mult")
sess = tf.Session()
res = sess.run(mult)

print("matrix product = ", res)

writer = tf.summary.FileWriter("./m2_simplemath", sess.graph)
writer.close()
sess.close()
