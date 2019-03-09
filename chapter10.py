
import tensorflow as tf
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
#
# Set up
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
# Training NN using plain TensorFlow
#
# Construction phase

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

# Placeholders for training data and targets.
reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X") # The number of training examples is unknown (hence the shape is None)
y = tf.placeholder(tf.int32, shape=(None), name="y")

# Neuron layer
def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):             # Name scope
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev) # will speed-up training (Xavier initialization)
        W = tf.Variable(init, name="kernel") # Weight matrix (aka kernel)
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z
# Deep NN - three layers
with tf.name_scope("dnn"):
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1",
                           activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2",
                           activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs")

# TensorFlow comes with many handy functions to create standard neural networks layers, so there is often no need to define
# your own neural_layer() function.
# tf.layers.dense() : creates fully connected layer where all the inputs are connected to all the neurons in the layer. It takes
# care of creating the weights and biases variables (named kernel and bias respectively), using the appropriate initialization
# strategy. The activation function can be set by using the activation argument :

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
    y_proba = tf.nn.softmax(logits)
# Cost function - sparse_softmax_cross_entropy_with_logits() - it computes the cross entropy based on the "logits" (the output
# of the network BEFORE going through softmax activation function) and it expects labels in the form of integers ranging from 0
# to the number of classes minus 1 .
# This function is equivalent to applying softmax activation function and then computing the cross entropy, but it is more
# efficient and it properly takes care of corner cases like logits equal to 0.  Note that there is another function called
# softmax_cross_entropy_with_logits(), which takes labels in the form of one-hot vectors

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

# Training
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
#
# The last step in the construction phase is to specify how to evaluate the model. We will use accuracy as our performance
# measure. For each instance, determine if the neural network's prediction is correct by checking whether or not the highest
# logit corresponds to the target class.
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)                    # in_top_k returns batch_size bolean array (the element i is True
                                                            # if the prediction for the target class is among the top k predictions
                                                            # among all predictions for example i)
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))  # Cast the vector 'correct' (boolean) to float and compute the average

# Create a node to initialize all the variables

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Execution phase
n_epochs = 20
n_batches = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch}) # Training set error
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid}) # validation set error
        print(epoch, "Batch accuracy:", acc_batch, "Validation accuracy:", acc_valid)

    save_path = saver.save(sess, "./my_model_final.ckpt")


# The NN is trained. The next step is to make predictions
with tf.Session() as sess:
    saver.restore(sess, "./my_model_final.ckpt") # or better, use save_path
    X_new_scaled = X_test[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled}) # evaluates logits node.
    y_pred = np.argmax(Z, axis=1)                # if you wanted to know all the estimated class probabilities, you would need to
                                                 # apply the softmax() function to the logits, but if you just want to predic
                                                 # a class, you can simply pick the class that has the highest logit value

# Hyper parameter tuning :
# randomized search (chapter 2)
# Oscar (http://oscar.calldesk.ai)
