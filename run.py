# This file retrieves the data, creates an algorithm, runs the prediction, saves the results to files, then creates plots of the error
import numpy as np
from mlp import MLP
from load_data import data_load

def main():
  net = MLP(124)
  for i in range(1, 75):
    data = data_load('13A-no_0',i-1,i)
    data = data.astype(int)
    net.train(data, 25)
  
  for i in range(75, 100):
    test = data_load('13A-no_0',i-1,i)
    test = test.astype(int)
    res = net.test(test[:,0], 201)
    res = res.astype(int)
    np.savetxt("results/test25epochs2hn_" + `i - 1` +  "_actual.csv", test, delimiter=",", fmt="%d")
    np.savetxt("results/test25epochs2hn_" + `i - 1` +  "_predicted.csv", res, delimiter=",", fmt="%d")

main()      

