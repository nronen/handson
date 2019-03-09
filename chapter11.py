#
# by default, tf.layers.dense() function uses Xavier initialization (with a uniform distribution). You can change this to He
# initialization by using the variance_scaling_initializer() function like this :
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

he_init = tf.variance_scaling_initializer()
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu,
                          kernel_initializer=he_init, name="hidden1")

# He initialization (Table 11-1), considers only fan-in not the average between fan-in and fan-out like in Xavier initialization.
# This is also the default for the variance_scaling_initializer() function, but you can change this by setting the argument
# mode="FAN_AVG"
#
# Relu: It turns out that activation functions other than sigmoid behave much better in deep neural networks, in particular ReLU
# activation function, mostly because it does not saturate for positive values (and also becuase it is quite fast to compute).
# The ReLU functon suffers from a problem known as dying ReLUs : during training, some neurons effectively die, meaning they
# stop outputting anything other than 0 -- during training, if a neuron's weights get updated such that the weighted sum of the
# neuron's input is negative, it will start outputting 0. When this happen, the neuron is unlikely to come back to life since
# the gradient of the ReLU function is 0 when its input is negative.
#
# ELU -- (z < 0): a(exp(z)-1) ; (z>=0): z
# To solve this you may want to use a varient of the ReLU function such as leaky ReLU. LeakyRelu(a) = max(az,z
# In general ELU > leakyRelu (and its variants) >  ReLU > tanh > sigmoid . If you care a lot about runtime performance, then you
# may prefer leakyRelu over ELUs. If you don't want to tweak another hyperparameter, you may just use the default a values (a=0.01
# for the leakyRelu , a=1 for ELU)
# leaky ReLU :

def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

hidden1 = tf.layers.dense(X, n_hidden1, activation=leaky_relu, name="hidden1")

# ELU
hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.elu, name="hidden1")

# SELU
# During training, a neural network composed of a stack of dense layers using the SELU activation function will self-normalize:
# the output of each layer will tend to preserve the same mean and variance during training, which solves the vanishing/exploding
# gradients problem. As a result, this activation function outperforms the other activation functions very significantly for such
# neural nets
#
# By default, the SELU hyperparameters (scale and alpha) are tuned in such a way that the mean remains close to 0, and the
# standard deviation remains close to 1 (assuming the inputs are standardized with mean 0 and standard deviation 1 too).
# The tf.nn.selu() function was added in TensorFlow 1.4. For earlier versions, you can use the following implementation:
def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    return scale * tf.where(z >= 0.0, z, alpha * tf.nn.elu(z))
# However, the SELU activation function cannot be used along with regular Dropout (this would cancel the SELU activation function's
# self-normalizing property). Fortunately, there is a Dropout variant called Alpha Dropout proposed in the same paper. It is available
# in tf.contrib.nn.alpha_dropout() since TF 1.4

# Batch Normalization (Batch Norm.)
# Tensorflow provides a tf.nn.batch_normalization() funciton that simply centers and normalizes the inputs, but you must compute
# the mean and standard deviation by yourself (based on the mini-batch data during training or on the full data set during testing)
# and pass them as parameters to this function and you must also handle the creation of the scaling and offset parameters (and pass
# them to this function). It is doable but not the most convinient approach. Instead, you should use the
# tf.layers.batch_normalization() function, which handles all this for you.
reset_graph()

import tensorflow as tf

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")

# training will be set to True only during training --> this is needed to tell the tf.layers.batch_normalization() function whether
# it should use the current mini-batch's mean and standard deviation (during training) or the whole training set's mean and
# standard deviation (during testing)
training = tf.placeholder_with_default(False, shape=(), name='training')

# BN uses   exponential decay to compute the running average (hence the momentum parameter is definec)
hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)

hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)

logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = tf.layers.batch_normalization(logits_before_bn, training=training,
                                       momentum=0.9)

# To avoid repeating the same parameters over and over again, we can use Python's partial() function.
# neural net for MNIST, using the ELU activation function and Batch Normalization at each layer:
from functools import partial

reset_graph()

batch_norm_momentum = 0.9

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name='training')

with tf.name_scope("dnn"):
    he_init = tf.variance_scaling_initializer()

    my_batch_norm_layer = partial(
            tf.layers.batch_normalization,
            training=training,
            momentum=batch_norm_momentum)

    my_dense_layer = partial(
            tf.layers.dense,
            kernel_initializer=he_init)

    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    bn1 = tf.nn.elu(my_batch_norm_layer(hidden1))
    hidden2 = my_dense_layer(bn1, n_hidden2, name="hidden2")
    bn2 = tf.nn.elu(my_batch_norm_layer(hidden2))
    logits_before_bn = my_dense_layer(bn2, n_outputs, name="outputs")
    logits = my_batch_norm_layer(logits_before_bn)

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# execution phase - same as before with two exceptions :
# 1. During training, whenever you run an operation that depends on the batch_normalization() layer, you need to set
#    the training placeholder to True
# 2. The batch_normalization() function creates a few operations that must be evaluated at each step during training in order to
#    update the moving average. These operations are automatically added to UPDATE_OPS collection, so all we need to do is
#    to get the list of operations in that collection and run then at each training iteration.
n_epochs = 20
batch_size = 200

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run([training_op, extra_update_ops],
                     feed_dict={training: True, X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")

# Gradient Clipping
# A popular tehcnique to lessen the exploding gradients problem is to simply clip the gradients during backprop. so that
# they never exceed some threshold (in general people now prefer BN):
# In TF, the optimizer's minimize() function takes care of both computing the gradients and applying them, so you must
# instead call the optimizer's compute_gradients() method first , then create an operation to clip the gradients using
# clip_by_value() function, and finally create an operation to apply the clipped gradients using the optimizer's
# apply_gradients() function

threshold = 1.0

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
              for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)

# L1 regulrization
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu, name="hidden1")
    logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

# Next, we get a handle on the layer weights, and we compute the total loss, which is equal to the sum of the usual cross
# entropy loss and the l1 loss (i.e., the absolute values of the weights):
W1 = tf.get_default_graph().get_tensor_by_name("hidden1/kernel:0")
W2 = tf.get_default_graph().get_tensor_by_name("outputs/kernel:0")

scale = 0.001 # l1 regularization hyperparameter

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
    reg_losses = tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2))
    loss = tf.add(base_loss, scale * reg_losses, name="loss")

# The rest is just as usual:
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 200

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
# If there are many layers, this approach is not very convinient. TF provides a better option : many functions
# that create variables access a *_regularizer argument for each created variable (e.g. kernel_regularizer)
reset_graph()

n_inputs = 28 * 28  # MNIST
n_hidden1 = 300
n_hidden2 = 50
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

scale = 0.001

my_dense_layer = partial(
    tf.layers.dense, activation=tf.nn.relu,
    kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("dnn"):
    hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
    hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
    logits = my_dense_layer(hidden2, n_outputs, activation=None, name="outputs")

# Next we must add the regularization losses to the base loss:
with tf.name_scope("loss"):                                     # not shown in the book
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(  # not shown
        labels=y, logits=logits)                                # not shown
    base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")   # not shown
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([base_loss] + reg_losses, name="loss")

# And the rest is the same

# Dropout
# The most popular technique for deep NN is agruably dropout -- Neurons trained with dropout, cannot rely exclusively on
# just a few input neurons; they must pay attention to each of their input neurons. They end up being less sensetive to
# slight changes in the inputs. In the end you get a more robust network that generalizes better.

# Another way to understand the power of dropout is to realize that a unique NN is generated at each training step. Since
# each neuron can be either present or absent, there is a total of 2**N possible networks (N is the total number of
# droppable neurons). These NN are obviously not independent since they share many of their weights, but they are
# nevertheless all different. The resulting network can be seen as an averaging ensamble of all of these smalller NN.

# Suppose p=50% , in which case during testing a neuron will be connected to twice as many input neurons as it was (on
# average) during training. To compensate for this fact, we need to multiply each neuron's input connection by 0.5 after
# training. If we don't , each neuron will get a total input signal roughly twice as large as what the network was trained
# on, and it is unlikely to perform well. More generally we need to multiply each input connection weight by the
# keep-probability (1-p) after training.

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

training = tf.placeholder_with_default(False, shape=(), name='training')

dropout_rate = 0.5  # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu,
                              name="hidden1")
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu,
                              name="hidden2")
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./my_model_final.ckpt")
