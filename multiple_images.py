import tensorflow as tf
import numpy as np
import matplotlib.image as mp_img
import matplotlib.pyplot as plt
from PIL import Image

import os

original_image_list = ["./flower01.jpg",
                       "./dogs01.jpg",
                       "./cat01.jpg",
                       "./ball.jpg"]

filename_queue = tf.train.string_input_producer(original_image_list)

image_reader = tf.WholeFileReader()

with tf.Session() as sess:
    # Coordinate the loading of image files
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_list = [];
    for i in range(len(original_image_list)):
        # Read a whole file from the queue, the first returned value in the tuple is th
        # filename which we are ignoring.
        _, image_file = image_reader.read(filename_queue)

        # Decode the image as a JPEG file, this will turn it into a Tensor which we can
        # then use in training
        image = tf.image.decode_jpeg(image_file)

        # get a Tensor of resized images.
        image = tf.image.resize_images(image, [224, 224])
        image.set_shape((224, 224, 3))

        # Get an image tensor and print its value.
        image_array = sess.run(image)
        print("{} : {}".format(image_file,image_array.shape))


        Image.fromarray(image_array.astype('uint8'), 'RGB').show()

        # The expand_dims adds a new dimension
        image_list.append(tf.expand_dims(image_array, 0))

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)

    images_tensor = tf.stack(image_list)
    print(images_tensor)

    #Write image summary
    summary_writer = tf.summary.FileWriter('./debug_multiple_images', graph=sess.graph)

    summary_str = sess.run(tf.summary.image("images", images_tensor))
    summary_writer.add_summary(summary_str)

    summary_writer.close()

