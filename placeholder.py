import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.int32, shape=[3], name='x')
y = tf.placeholder(tf.int32, shape=[3], name='y')

sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")

final_div = tf.div(sum_x, prod_y, name="final_div")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

sess = tf.Session()

print("sum(x) = ", sess.run(sum_x, feed_dict={x: [100, 200, 300]}))
print("prod(y) = ", sess.run(prod_y, feed_dict={y: [1, 2, 4]}))

print("final div = ", sess.run(final_div, feed_dict={x: [10, 20, 30], y: [1, 2, 3]}))
print("final mean = ", sess.run(final_mean, feed_dict={x: [1000, 2000, 3000], y: [10, 20, 30]}))


writer = tf.summary.FileWriter("./m2_simplemath", sess.graph)
writer.close()
sess.close()
