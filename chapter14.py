# Common imports
import tensorflow as tf
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


# Basic RNN - a layer of 5 recurrent neurons, using tanh activation function . This RNN runs over two time steps , taking
# input vectors of size 3 at each time step
reset_graph()

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons,n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

init = tf.global_variables_initializer()

X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
    init.run()
    Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})

# Static unrolling through time. The static_rnn() function creates an unrolled RNN network by chaining cells. The following code
# creates the exact same model as the previous one

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [X0, X1],
                                                dtype=tf.float32)
Y0, Y1 = output_seqs

# If there were 50 time steps, it would not be very convinient to have to define 50 input placeholders and 50 output tensors
# Moreover, at execution time you would have to define 50 input placeholders and manipulate the 50 outputs. The following code
# builds the same RNN again, but this time it takes a single input placeholder of shape [none, n_steps, n_inputs ] where the first
# dimension is the mini-batch size, then it extracts the list of input sequences for each time step.

# X_seq is a Python list of n_steps tensors of shape [none, n_inputs]. To do this , we first swap the first two dimensions using
# the transpose() functions so that the time steps are now the first dimension. Then we extract a Python list of Tensors
# along the first dimension (i.e. one tensor per time step) using the unstack() function.

# Finally, we merge all the output tensors using the stack() function, and we swap the first two dimension to get final outputs
# tensor of shape [None, n_steps, n_neurons]


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs,
                                                dtype=tf.float32)
outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

# With such a large graph you may even get out-of-memory (OOM) errors during backpropagation, since it must store all tensor values
# during forward pass to it can use them to compute gradients during the reverse pass.  The solution is to yse dynamic_rnn() function
# This function uses a while_loop() operation to run over the cell the appropriate number of times, and you can set swap_memory=True
# if you want it to swap the GPU's memory to the CPU's memory during backpropagation to avoid OOM errors

# Conveniently, it also accepts a single tensor for all inputs at every time step (shape [None, n_steps, n_inputs]) and it outputs
# a single tensor for all outputs at every time step (shape [None, n_steps, n_inputs]); there is no need to stack, unstack or
# transpose:

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# Handling variable length input sequences.
# If the input sequences have variable lengths - then you should set the sequence length argument when calling the dynamic_rnn()
# (or static_rnn()) function. It must be 1D tensor indicating the length of the input sequence for each instance :
n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

seq_length = tf.placeholder(tf.int32, [None])
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                    sequence_length=seq_length)

init = tf.global_variables_initializer()

# The second input must be padded with zeros in order to fit the input tensor X
X_batch = np.array([
        # step 0     step 1
        [[0, 1, 2], [9, 8, 7]], # instance 1
        [[3, 4, 5], [0, 0, 0]], # instance 2 (padded with zero vectors)
        [[6, 7, 8], [6, 5, 4]], # instance 3
        [[9, 0, 1], [3, 2, 1]], # instance 4
    ])
seq_length_batch = np.array([2, 1, 2, 2])

with tf.Session() as sess:
    init.run()
    outputs_val, states_val = sess.run(
        [outputs, states], feed_dict={X: X_batch, seq_length: seq_length_batch})

# The RNN outputs zero vectors for every time step past the input sequence length :
# print(outputs_val)
#[[[-0.9123188   0.16516446  0.5548655  -0.39159346  0.20846416]
#  [-1.          0.956726    0.99831694  0.99970174  0.96518576]]  # final state
#
# [[-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]   # final state
#  [ 0.          0.          0.          0.          0.        ]]  # zero vector
#
# [[-0.99999976  0.8967997   0.9986295   0.9647514   0.93662   ]
#  [-0.9999526   0.9681953   0.96002865  0.98706263  0.85459226]]  # final state
#
# [[-0.96435434  0.99501586 -0.36150697  0.9983378   0.999497  ]
#  [-0.9613586   0.9568762   0.7132288   0.97729224 -0.0958299 ]]] # final state

# The states tensor contains the final state of each cell (excluding the zero vectors):
# print(states_val)
# [[-1.          0.956726    0.99831694  0.99970174  0.96518576]   # t = 1
#  [-0.9998612   0.6702289   0.9723653   0.6631046   0.74457586]   # t = 0 !!!
#  [-0.9999526   0.9681953   0.96002865  0.98706263  0.85459226]   # t = 1
#  [-0.9613586   0.9568762   0.7132288   0.97729224 -0.0958299 ]]  # t = 1

# Handling variable length output sentences -- the most common solution is to define EOS (end-of-sequence) token. Any output
# past EOS should be ignored

# Training a sequence classifier
# Each image is treated as a sequence of 28 rows of 28 pixels each. Each RNN cell has 150 neurons. A fully connected layer
# containing 10 neurons is connected to the output of the last time step. Note that the fully connected layer is connected to
# the states tensor , which contains only the final state of the RNN (i.e. the 28th output)
reset_graph()

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                          logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

# You can specifiy an initializer to the RNN by wrapping its construction code in a variable scope (i.e. use variable_scope("rnn"),
# initializer=variance_scaling_initializer()) to use He initialization)

# Improving accuracy :
# - tuning the hyperparameters
# - initializing the RNN weights using He initialization
# - training longer
# - adding a bit of regularization (e.g. dropout)

# Training to predict time series
