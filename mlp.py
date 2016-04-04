import numpy as np
import time
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, TanhLayer, FullConnection, GaussianLayer, SigmoidLayer
from pybrain.datasets import SupervisedDataSet

# To test, create array:
# 
# import numpy as np
# a = np.array([[ 1,  0,  0,  1,  1,  1,  0,  0], [ 0,  0,  1,  0,  1,  1,  0,  1], [ 0,  0,  1,  1,  1,  0,  0,  1]])
# from mlp import MLP
# net = MLP(3)
# net.train(a,3)
# net.test((1,0,0),8)

class MLP:
  def __init__(self, numMolecules):
    self.nm = numMolecules
    network = buildNetwork(numMolecules, numMolecules, numMolecules, hiddenclass=TanhLayer, outclass=SigmoidLayer)
    self.network = network

  def train(self, input, epochs):
    nm = self.nm
    ds = SupervisedDataSet(nm, nm)
    ds.setField('input', np.transpose(np.delete(input,0,1)))
    ds.setField('target', np.transpose(np.delete(input,input.shape[1]-1,1)))
    trainer = BackpropTrainer(self.network, ds)
    start = time.time()
    for index in range(epochs):
      print "Running epoch " + `index` + "..."
      err = trainer.train()
      print "Epoch resulted with " + `err` + " error."
    elapsed = (time.time() - start)
    print "Took " + `elapsed` + "ms to run."

  def test(self, input, ts):
    net = self.network
    res = np.zeros((self.nm, ts))
    res[:,0] = input
    for i in range(1,ts-1):
      tmp = net.activate(res[:,i-1])
      tmp[tmp>0.6] = 1
      tmp[tmp<=0.6] = 0
      res[:,i] = tmp
    return res
