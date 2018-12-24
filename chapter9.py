# the code just create a computation graph ( the variables are not initialized ).
import tensorflow as tf

x = tf.Variable(3,name="x")
y = tf.Variable(4,name="y")
f = x*x*y + y + 2

# The evaluate this graph, you need to open a TensorFlow session and to use it to initialize the variables,
# and evaluate f. The following code creates a session, initializes the variables, evalutes f and then closes the session

sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
print(result)
sess.close()

# A better way :

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

# Inside the with block,the session is set as the default session
# Calling x.initializer.run() is equivalent to calling tf.get_default_session.run(x.initializer) and similarly f.eval() is
# equivalent to tf.get_default_session.run(f)
# The session is automatically closed at the end of the block

# Instead of manually running the initializer for every single variable, you can use the function
# global_variable_initalizer(). Note that it does not actually perform the initialization immediately but rather creates
# a node in the graph that will initialize all the variable when it is run:

init = tf.global_variable_initalizer()
with tf.Session() as sess:
    init.run()
    result = f.eval()

# Any node you create is automatically added to the default graph:
x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

# Sometimes you may want to manage multiple independent graphs. You can do this by creating a new Graph and temporarily making
# it the default graph inside a with block:
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph
x2.graph is tf.get_default_graph()
# it is common to run the same commands more than once while you are experimenting. As a result, you may end up with a as_default
# graph containing many duplicate nodes. One solution is to reset the default graph by running tf.reset_default_graph()


# When you evaluate a note , TensorFlow automatically determines the set of nodes that it depends on and it evaluates these nodes
# first.

w = tf.constant(3)
x = w + 2
y = x + 5
z = y + 3

with tf.Session as sess:
    print(y.eval())
    # TF detects that y depends on x , which depends on w, so it first evaluates w, then x and then y
    print(z.eval())

# All nodes values are dropped between graph runs, except variable values which are maintained by the session across graph runs
# A variable starts its life when its initializer is run and it ends when the session is closed

# If you want to evaluate y and z efficiently,without evaluating w and x twice as in the previous code, you must ask TF
# to evaluate both y and z in just one graph run:
with tf.Session as sess:
    y_val, z_val = sess.run([y,z])
    print(y_val)
    print(z_val)

# TF operations (aka ops) can take any number of inputs and produce any number of outputs. Constants and variables take no
# inputs (they are called source ops). The inputs and outputs are multi-dimesional arrays called Tensors (hence the name
# TensorFlow). In python API, tensors are simply represented by NumPy ndarrays.

# Linear regression -- using Normal equation
import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m,n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones(m,1) , housing.data]
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1)), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

with tf.Session as sess:
    theta_value = theta.eval()

# pure Numpy :
X = housing_data_plus_bias
y = housing.target.reshape(-1, 1)
theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

print(theta_numpy)

# Scikit learn :
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing.data, housing.target.reshape(-1, 1))

print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

# When using gradient descent , remember that it is important to first normalize the input feature vectors , or else training
# may be much slower (you can do this by using TensorFlow, NumPy, Sci-Kit learn's StandardScaler or any other solution)

# Gradient Descent requires scaling the feature vectors first. We could do this using TF, but let's just use Scikit-Learn
# for now.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 1000
learning_rate = 0.01

X.tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
y.tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y"))
# random_uniform() creates a node in the graph that will generate a tensor containing random values
theha = tf.Variable(rf.random_uniform([n + 1, 1], 1.0, 1.0, seed=42, name=theta))
y_pred = tf.matmul(X, theta, name="prediction")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error),name="mse")
gradients = 2/m * tf.matmul(tf.transpose(X),error)
# assign() creates a node that will assign a new value to a variable. In this case, it implements Batch Gradient descent step
training_op = tf.assign(theta, theta - learning_rate * gradients)

init = global_variable_initalizer()

with tf.Session as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if (epoch % 100) == 0 :
            print("Epoch",epoch, "MSE=".mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()

# autodiff -- it can automatically and efficiently compute the gradients for you.

# The gradients function takes an op(in this case mse), and a list of variables (in this case theta), and it creates
# a list of ops (one per variable) to compute the gradients of the op with regards to each variable. The following code
# computes the gradients vector of the MSE with regards to theta

gradients = tf.gradients(mse, [theta])[0]

# TF provides a number of optimizers out of the box :
optimizer = tf.train.GradientDescentOptimizer(learnig_rate = learning_rate)
training_op = optimizer.minimize(mse)

# a different optimizer (w/ momentum) :
optimizer = tf.train.GradientDescentOptimizer(learnig_rate = learning_rate, momentum=0.9)

# Let's try to modify the previous code to implement Mini-Batch Gradient Descent. For this we will need a way to replace x and y at
# every iteration with tthe next mini-batch. The simplest way to do that is to use placeholder nodes. These nodes are special
# because they don't actually perform any computation, they just output the data you tell output at runtime.

# To create a placeholder, you must call the placeholder() function and specify the output tensor's data type. Optionally , you can
# also specify its shape, if you want to enforce it. If you specify None for a dimension, it means "any size"

# The following code creates a placeholder A and a node B
A = tf.placeholder(tf.float32, shape=(None,3))
B = A + 5
# When we evaluate B, we pass feed_dict to eval() method that specifies the value of A :
with tf.Session() as sess:
    B_val_1 = B.eval(feed_dict={A: [[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A: [[1,2,3],[4,5,6]]})
print (B_val_1)
print (B_val_2)

# Note that you can feed the output of any operation , not just placeholders.

# To implement mini-batch gradient descent , we only need to tweak the existing code slightly: the definition of x,y in the
# construction phase is changed to make then placeholder nodes:
X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

# Mini-batch Gradient Descent
n_epochs = 10
learning_rate = 0.01

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

batch_size = 100
n_batches = int(np.ceil(m / batch_size))

def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
    indices = np.random.randint(m, size=batch_size)  # not shown
    X_batch = scaled_housing_data_plus_bias[indices] # not shown
    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()

# Once you have trained your model, you should save its parameters to disk so you can come back to it whenever you
# want , use it another program, compare it to other models, and so on. Moreever, you probably want to save checkpoints
# at regular intervals during training so that if your computer crashes during training you can continue from the last
# checkpoint rather than start over from scratch

# Create a Saver node at the end of the construction phase (after all the variables are created);  then in the execution
# phase just call its save() method

# Batch Gradient Descent
reset_graph()

n_epochs = 1000                                                                       # not shown in the book
learning_rate = 0.01                                                                  # not shown

X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")            # not shown
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")            # not shown
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")                                      # not shown
error = y_pred - y                                                                    # not shown
mse = tf.reduce_mean(tf.square(error), name="mse")                                    # not shown
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            # not shown
training_op = optimizer.minimize(mse)                                                 # not shown

# The optimizer will modify the value of theta in order to minimize the loss (mse)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE =", mse.eval())                                # not shown
            save_path = saver.save(sess, "/tmp/my_model.ckpt")
        sess.run(training_op)

    best_theta = theta.eval()
    save_path = saver.save(sess, "/tmp/my_model_final.ckpt")

# Restoring the model: you create a Saver node at the end of the construction phase (as before). but then
# instead of initializing the variables using the init node, you call the restore() method of the Saver project:
with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval() # not shown in the book

# By default a Saver saves and restores all variables under their own name, but if you need more control, you can specify which
# variables to save or restore and what names to use. For example, the following Saver will save or restore only the theta
# variable under the name weights
saver = tf.train.Saver({"weights": theta})

# By default the saver also saves the graph structure itself in a second file with the extension .meta. You can use the function
#tf.train.import_meta_graph() to restore the graph structure. This function loads the graph into the default graph and returns a
# Saver that can then be used to restore the graph state (i.e., the variable values):
reset_graph()
# notice that we start with an empty graph.

saver = tf.train.import_meta_graph("/tmp/my_model_final.ckpt.meta")  # this loads the graph structure
theta = tf.get_default_graph().get_tensor_by_name("theta:0") # not shown in the book

with tf.Session() as sess:
    saver.restore(sess, "/tmp/my_model_final.ckpt")  # this restores the graph's state
    best_theta_restored = theta.eval() # not shown in the book

# Visualizing the graph and training curves using TensorBoard
# The first step is to tweak your program a bit so it writes the graph definition and some training stats to a log directory
# that TensorBoard will read from. You need to use a different log directory every time you run your program, or else
# TensorBoard will merge stats from different runs, which will mess up the visualizations. The simplest solution for this is
# to include a timestamp in the log directory name :
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

# Next add the following code at every end of the construction phase
#
# create a node in the graph that will evaluate the MSE value and write it to a TensorBoard compatible binary log string
# called a summary.
mse_summary = tf.summary.scalar('MSE', mse)
# Create a FileWriter that you will use to write summaries to the logfiles in the log directory. Upon creation the FileWrite
# creates the log directoty (if does not already exists) and writes the graph definition to a binary logfile called an
# events file.
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# Next you need to update the execution phase to evaluate the mse_summary node regulary during training (e.g. every 10 min
# batches). Here's the updated code :

reset_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

n_epochs = 1000
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

n_epochs = 10
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

with tf.Session() as sess:                                                        # not shown in the book
    sess.run(init)                                                                # not shown

    for epoch in range(n_epochs):                                                 # not shown
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

    best_theta = theta.eval()
file_writer.close()                                                  # not shown

#
# The logs are written to tf_logs/run*
# start TensorBoard server  :
# tensorboard --logdir tf_logs/
# The next step is to open browser and go to http://0.0.0.0:6006/

# When dealing with more complex models sucn as NN, the graph can easily become cluttered with thousands of nodes. To avoid this
# you can create name spaces to group related nodes :

# The previous code is modified to define the error and mse ops within a name scope called "loss"
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

# Modularity
# An ugly flat code the computes the Relu function multiple times :
reset_graph()

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0., name="relu1")
relu2 = tf.maximum(z1, 0., name="relu2")  # Oops, cut&paste error! Did you spot it?

output = tf.add(relu1, relu2, name="output")

# A more modular code , using a new function :

reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
        b = tf.Variable(0.0, name="bias")                             # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
        return tf.maximum(z, 0., name="max")                          # not shown

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
file_writer.close()

# Sharing a threshold variable the classic way, by defining it outside of the relu() function then passing it as a parameter:
reset_graph()

def relu(X, threshold):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

threshold = tf.Variable(0.0, name="threshold")
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X, threshold) for i in range(5)]
output = tf.add_n(relus, name="output")

# If there are many shared parameters, it will be painful to have to pass them around as parameters all the time.
# Other alternatives
# 1. Using Python dictionaries containing all the variables in the model
# 2. Creating a class for each module
# 3. Set the shared variable as an attribute of the relu function
reset_graph()

def relu(X):
    with tf.name_scope("relu"):
        if not hasattr(relu, "threshold"):
            relu.threshold = tf.Variable(0.0, name="threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, relu.threshold, name="max")
# 4. Use the get_variable() function to create the shared variable (if does not exist yet), or reuse it if already exists.
# The desired behavior is controlled by an attribute of the current variable variable_scope()

with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
# If the variable has already been created by an earlier call to get_variable(), this code will raise an exception. The behavior
# prevents resuing variables by mistake. If you want to reuse a variable, you need to explicitly say so by setting the variable's
# scope resue to True (in which case you don't have to specify the shape or the initializer)

# This code will fetch the existing "relu/threshold" variable, or raise an exception if it does not exist of if it was not created
# using get_variable().
with tf.variable_scope("relu", reuse=True):
    threshold = tf.get_variable("threshold")

# Alternatively, you can set the reuse attribute to True inside the block by calling the scope's reuse_variables() method :
with tf.variable_scope("relu") as scope:
    scope.reuse_variables()
    threshold = tf.get_variable("threshold")

# The threshold is created once (outside the Relu function) and then it is reused inside this function:
reset_graph()

def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = int(X.get_shape()[1]), 1                          # not shown
        w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
        b = tf.Variable(0.0, name="bias")                           # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
        return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for relu_index in range(5)]
output = tf.add_n(relus, name="output")

# The threshold variable is defined outside the Relu function ,where all the rest of ReLU code resides. To fix this the following
# code creates the threshold variable within the relu() function upon the first call, then reuses it in subsequent calls :

reset_graph()

def relu(X):
    threshold = tf.get_variable("threshold", shape=(),
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)                        # not shown in the book
    w = tf.Variable(tf.random_normal(w_shape), name="weights")  # not shown
    b = tf.Variable(0.0, name="bias")                           # not shown
    z = tf.add(tf.matmul(X, w), b, name="z")                    # not shown
    return tf.maximum(z, threshold, name="max")

X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for relu_index in range(5):
    with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")
