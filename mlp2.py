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

class MLP:
  def __init__(self, nm, hn):
    self.nm = nm

    # Hidden layer: Sigmoidal
    x = tf.placeholder(tf.float32, [None, nm])
    W_hidden = tf.Variable(tf.zeros([nm, hn]))
    b_hidden = tf.Variable(tf.zeros([hn]))
    y_hidden = tf.nn.sigmoid(tf.matmul(x, W_hidden) + b_hidden)

    # Output layer: Softmax
    W_output = tf.Variable(tf.zeros([hn, nm]))
    b_output = tf.Variable(tf.zeros([nm]))
    y = tf.nn.sigmoid(tf.matmul(y_hidden, W_output) + b_output)

    self.net = y

  def train(self, input, epochs):
    net = self.net

    # Trainer setup
    y_ = tf.placeholder(tf.float32, [None, nm])
    cross_entropy = -tf.reduce_sum(y_*tf.log(net))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    
    # Initialize
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    self.train_step = train_step

    x = np.transpose(np.delete(input,0,1))
    y = np.transpose(np.delete(input,input.shape[1]-1,1))
    start = time.time()
    for epoch in range(epochs):
        sess.run(train_step, feed_dict={x: x, y_: y})
        # Run error here
        print "Epoch resulted with " + `err` + " error."
    elapsed = (time.time() - start)
    print "Took " + `elapsed` + "ms to run."

  def test(self, input, ts):
    net = self.net
    # Initialize
    sess = tf.Session()
    res = np.zeros((self.nm, ts))
    res[:,0] = input
    for i in range(1,ts-1):
      tmp = sess.run(y, feed_dict={x: res[:,i-1]}
      tmp[tmp>0.55] = 1
      tmp[tmp<=0.55] = 0
      res[:,i] = tmp
    return res
