import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, TanhLayer, FullConnection
from pybrain.datasets import SupervisedDataSet

class MLP:
  def __init__(self, numMolecules):
    self.nm = numMolecules
    network = buildNetwork(numMolecules, numMolecules, numMolecules, hiddenclass=TanhLayer)
    self.network = network

  def train(self, input, epochs):
    nm = self.nm
    ds = SupervisedDataSet(nm, nm)
    ds.setField('input', np.delete(input,0,1))
    ds.setField('target', np.delete(input,input.shape[1],1))
    trainer = BackpropTrainer(self.network, dataset)
    for index in range(epochs):
      trainer.train()

  def test(self, input, ts):
    net = self.network
    res = zeros(self.nm, ts)
    res(:,0) = input
    for i in range(2,ts-1):
      res(:,i) = net.activate(res(:,i-1))
    return res
