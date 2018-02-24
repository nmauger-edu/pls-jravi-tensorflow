import numpy as np
import tensorflow as tf

# see also : https://github.com/aymericdamien/TensorFlow-Examples

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# Store the MNIST data in /tmp/data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

training_digits, training_labels = mnist.train.next_batch(50000)
test_digits, test_labels = mnist.test.next_batch(200)

# None because we don't know how many image in the list... ?!
training_digits_pl = tf.placeholder("float", [None, 784])
test_digits_pl = tf.placeholder("float", [784])

# Nearest neighbour
l1_distance = tf.abs(tf.add(training_digits_pl, tf.negative(test_digits_pl)))
distance = tf.reduce_sum(l1_distance, axis=1)

# Prediction: get the minimum distance index previously calculated
pred = tf.arg_min(distance, 0)

accuracy = 0

# Initialization of variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # loop over each value
    for i in range(len(test_digits)):
        # Get nearest neighbour
        nn_index = sess.run(pred, \
                            feed_dict={training_digits_pl: training_digits, test_digits_pl: test_digits[i, :]})

        # Get nearest neighbor class label and compare it with its true label
        print("Test", i, "Predicition:", np.argmax(training_labels[nn_index]),\
            "True Label:", np.argmax(test_labels[i]))

        # Calculate accuracy
        if np.argmax(training_labels[nn_index]) == np.argmax(test_labels[i]):
            accuracy += 1/len(test_digits)

    print("Done!")
    print("Accuracy:", accuracy)


