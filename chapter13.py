# CNN
# In TF, each input image is typically represented as a 3D tensor of shape [height, width , channels]. A mini-batch is represented
# as 4D tensor of shape [mini-batch size, height, width, channels]. The weights are represented as a 4D tensor of shape [height ,
# width, channels(n-1), channels]. The bias terms are simply represented as a 1D tensor of shape [channels]
#
# Convolutional layer

import tensorflow as tf
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Load two sample images using Scikit learn's load_sample_image()
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")
flower = load_sample_image("flower.jpg")
dataset = np.array([china, flower], dtype=np.float32)
batch_size, height, width, channels = dataset.shape

# Create 2 filters
filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # vertical line
filters[3, :, :, 1] = 1  # horizontal line

# Apply the two filters to both images using a convolutional layer built using TF tf.nn.conv2d() function
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))

# strides is a four element 1D array [batch stride, vertical stdide, horizontal stride , channel stride]
# batch stride - to skip some instances, channel stride - to skip some of the previous' layer feature maps or channels.
# Padding can be either "VALID" (i.e. no padding) or "SAME" (zero padding if neccesary )
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding="SAME")

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

# In a real CNN, you would let the training algorithm discover the best filters automatically. TF has a tf.layers.conv2d() function
# which creates the filters available for you (called kernel), and initializes it randomly :

reset_graph()

X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2],
                        padding="SAME")

# Tip
# if training crashes because of an out-of-memory error :
# - reduce the mini batch size
# - reduce the dimensionality using a stride
# - remove some layers
# - use 16 bit floats instead of 32-bit floats
# - distribute the CNN accross multiple devices

# implementing max pooling layer:
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))

# The ksize argument contains the kernal shape along all four dimensions of the input tensor [batch size - 0, height - 1, width - 2,
# channels - 3] TF does not support pooling over multiple instances, so the first element of ksize must equal to 1. Moreover, it
# does not support pooling over both the sapatial dimensions (height and width) and the depth dimentions, so either ksize[1]
# and ksize[2] must both be equal to 1, or ksize[3] (depth) must be equal to 1.

max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1],padding="VALID")

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})
