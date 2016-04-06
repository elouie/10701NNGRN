# This file retrieves the data, creates an algorithm, runs the prediction, saves the results to files, then creates plots of the error
import numpy as np
from mlp import MLP
from load_data import data_load

def main():
  numHiddenNodes = 124
  numMolecules = 124
  numTimesteps = 201
  numRuns = 100
  numEpochs = 100
  net = MLP(numMolecules, numHiddenNodes)

  # Load the data into a matrix for use over epochs
  data = zeros((numMolecules,numTimeSteps,numRuns))
  for i in range(numRuns):
    data[:,:,i] = data_load('13A-no_0',i,i+1).astype(int)

  # For each epoch
  for i in range(numEpochs)
    # For each run
    for j in range(75)
      net.train(data[:,:,j])
    # If we hit a quarterly epoch, run tests and output data and error
    if (i % 25 == 0)
      # Get train error and output
      for k in range(1, 75):
        res = net.test(data[:,0,k], 201)
        np.savetxt("results/data_hiddennodes" + `numHiddenNodes` + "_epochs" + `i` +  "_train_actual.csv", test, delimiter=",", fmt="%d")
      # Get test error and output
      for k in range(75, 100):
        res = net.test(data[:,0,k], 201)
        np.savetxt("results/data_hiddennodes" + `numHiddenNodes` + "_epochs" + `i` +  "_test_actual.csv", test, delimiter=",", fmt="%d")
  
main()      

