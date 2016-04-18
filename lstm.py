import numpy as np
import time
import tensorflow as tf

# To test, create array:
#
# import numpy as np
# a = np.array([[ 1,  0,  0,  1,  1,  1,  0,  0], [ 0,  0,  1,  0,  1,  1,  0,  1], [ 0,  0,  1,  1,  1,  0,  0,  1]])
# from mlp import MLP
# net = MLP(3,2)
# net.train(a,3)
# net.test((1,0,0),8)

class LSTM:
  # net = LSTM(nm, hn, hl)
  #   nm: Number of molecules in the data
  #   hn: Number of hidden nodes/units per layer
  #   hl: Number of hidden layers prior to the output layer
  #   net: Object representing the created network
  def __init__(self, nm, hn, hl):
    self.nm = nm

    # Hidden layer: Sigmoidal
    x = tf.placeholder(tf.float32, [None, nm])
    W_hidden = tf.Variable(tf.zeros([nm, hn]))
    b_hidden = tf.Variable(tf.zeros([hn]))
    y_hidden = tf.nn.sigmoid(tf.matmul(x, W_hidden) + b_hidden)

    # Output layer: Sigmoidal
    W_output = tf.Variable(tf.zeros([hn, nm]))
    b_output = tf.Variable(tf.zeros([nm]))
    y = tf.nn.sigmoid(tf.matmul(y_hidden, W_output) + b_output)

    self.net = y
    self.nm = nm

  def train(self, input, epochs):
    net = self.net
    nm = self.nm

    # Trainer setup
    y_ = tf.placeholder(tf.float32, [None, nm])
    cross_entropy = -tf.reduce_sum(y_*tf.log(net))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Initialize
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    self.train_step = train_step

    x = np.delete(input,0,1)
    y = np.delete(input,input.shape[1]-1,1)
    start = time.time()
    for epoch in range(epochs):
        for i in range(x.shape[1]):
          err = sess.run(train_step, feed_dict={x: x[:,i], y_: y[:,i]})
          # Run error here
          print "Epoch resulted with " + err + " error."
    elapsed = (time.time() - start)
    print "Took " + `elapsed` + "ms to run."

  def test(self, input, ts):
    net = self.net
    # Initialize
    sess = tf.Session()
    res = np.zeros((self.nm, ts))
    res[:,0] = input
    err = np.zeros((self.nm))

    x = np.transpose(np.delete(input,0,1))
    y = np.transpose(np.delete(input,input.shape[1]-1,1))

    for i in range(1,ts):
      tmp = sess.run(net, feed_dict={x: res[:,i-1].tolist()})
      print tmp
      tmp[tmp>0.5] = 1
      tmp[tmp<=0.5] = 0
      res[:,i] = tmp

    # Get error
    #correct_prediction =
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return res

  # save(fname)
  #   fname: A file location to save the network to
  #   Saves the network to a file for future use/training
  def save(fname):
    # TODO

  # net = load(fname)
  #   fname: A file location to save the network to
  #   Loads a network from a file and returns it
  def load(fname):
    # TODO
