import tensorflow as tf

hello = tf.constant('Hello, tensorflow')
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(6.5, name="constant_a")
b = tf.constant(3.4, name="constant_b")
c = tf.constant(3.0, name="constant_c")
d = tf.constant(100.2, name="constant_d")

square = tf.square(a, name="square_a")
power = tf.pow(b,c, name="pow_b_c")
sqrt = tf.sqrt(d, name="sqrt_d")

final_sum  = tf.add_n([square, power, sqrt], name="final_sum")

sess = tf.Session()

print("Square of a: ", sess.run(square))
print("Power of b^c: ", sess.run(power))
print("Square root of d: ", sess.run(sqrt))

print("Sum of square, power and square root : ", sess.run(final_sum))

another_sum = tf.add_n([a,b,c,d, power], name="another_sum")


x = tf.constant([100,200, 250], name='x')
y = tf.constant([1,2,10], name='y')


sum_x = tf.reduce_sum([x, y], name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")

final_div = tf.div(sum_x, prod_y, name="final_div")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

sess = tf.Session()

print("x: ", sess.run(x))
print("y: ", sess.run(y))
print("sum(x): ", sess.run(sum_x))
print("prod(y): ", sess.run(prod_y))
print("sum(x) / prod(y): ", sess.run(final_div))
print("mean(sum(x), prod(y): ", sess.run(final_mean))

writer = tf.summary.FileWriter("./m2_simplemath", sess.graph)
writer.close()
sess.close()
