# This file retrieves the data, creates an algorithm, runs the prediction, saves the results to files, then creates plots of the error
import numpy as np
from mlp import MLP
from load_data import data_load
from pybrain.tools.xml import NetworkWriter

def main():
  numHiddenNodes = 124
  numMolecules = 124
  numTimesteps = 201
  numRuns = 99
  numEpochs = 6
  net = MLP(numMolecules, numHiddenNodes)

  # Load the data into a matrix for use over epochs
  data = np.zeros((numMolecules,numTimesteps,numRuns))
  for i in range(numRuns):
    data[:,:,i] = data_load('13A-no_0',i,i+1).astype(int)

  # For each epoch
  for i in range(numEpochs):
    # For each run
    for j in range(75):
      net.train(data[:,:,j])
    # If we hit a quarterly epoch, run tests and output data and error
    if (i % 25 == 0):
      runString = `numHiddenNodes` + "_epochs" + `i` +  "_run"
      # Get train error and output
      trainErr = np.zeros(201)
      for k in range(1, 75):
        testData = data[:,0,k]
        fullTestData = data[:,:,k]
        res = net.test(testData, 201)
        err = net.meansqerr(fullTestData,res,201)
        trainErr = trainErr + err
        np.savetxt("results/data_hiddennodes" + runString + `k` + "_train_actual.csv", fullTestData, delimiter=",", fmt="%d")
        np.savetxt("results/data_hiddennodes" + runString + `k` + "_train_predicted.csv", res, delimiter=",", fmt="%d")
      np.savetxt("results/error_hiddennodes" + runString + `k` + "_train.csv", trainErr, delimiter=",", fmt="%f")

      trainErr = trainErr / 75
      # Get test error and output
      testErr = np.zeros(201)
      for k in range(75, numRuns):
        trainData = data[:,0,k]
        fullTrainData = data[:,:,k]
        res = net.test(trainData, 201)
        err = net.meansqerr(data[:,:,k],res,201)
        testErr = testErr + err
        np.savetxt("results/data_hiddennodes" + runString + `k` + "_test_actual.csv", fullTrainData, delimiter=",", fmt="%d")
        np.savetxt("results/data_hiddennodes" + runString + `k` + "_test_predicted.csv", res, delimiter=",", fmt="%d")
      testErr = testErr / 24
      np.savetxt("results/error_hiddennodes" + runString + `k` + "_test.csv", testErr, delimiter=",", fmt="%f")

      # Save the network thus far
      #NetworkWriter.writeToFile(net, "results/network_hiddennodes" + `numHiddenNodes` + "_epochs" + `i` +  "_run" + `k` + ".xml")
  
main()      

