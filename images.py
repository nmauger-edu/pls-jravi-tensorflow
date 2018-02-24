import tensorflow as tf
import numpy as np
import matplotlib.image as mp_img
import matplotlib.pyplot as plt
import os

filename = "./flower01.jpg"

image = mp_img.imread(filename)

print("Image shape: ", image.shape)
print("Image array: ", image)

plt.imshow(image)
plt.show()

x = tf.Variable(image, name='x')
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    #transpose = tf.transpose(x, perm=[1, 0, 2])  # swap first and second axis.
    # or same :
    transpose = tf.image.transpose_image(x)

    result = sess.run(transpose)

    print("Transposed image: ", result.shape)

    plt.imshow(result)
    plt.show()



