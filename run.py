# This file retrieves the data, creates an algorithm, runs the prediction, saves the results to files, then creates plots of the error
import numpy as np
import time
from mlp import MLP
from load_data import data_load
from pybrain.tools.customxml.networkwriter import NetworkWriter

def main():
  numHiddenNodes = 600
  numMolecules = 124
  numTimesteps = 201
  numRuns = 99
  numEpochs = 501
  net = MLP(numMolecules, numHiddenNodes)

  # Load the data into a matrix for use over epochs
  data = np.zeros((numMolecules,numTimesteps,numRuns))
  for i in range(numRuns):
    data[:,:,i] = data_load('13A-no_0',i,i+1).astype(int)

  # For each epoch
  for i in range(numEpochs):
    starttime = time.time()
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
        #np.savetxt("results/data_hiddennodes" + runString + `k` + "_train_actual.csv", fullTestData, delimiter=",", fmt="%d")
        if (k == 71):
          np.savetxt("results/data_hiddennodes" + runString + `k` + "_train_predicted.csv", res, delimiter=",", fmt="%d")
      trainErr = trainErr / 75
      print("Run " + `i` + " had " + `trainErr[200]` + " training error.");
      np.savetxt("results/error_hiddennodes" + runString + `k` + "_train.csv", trainErr, delimiter=",", fmt="%f")

      # Get test error and output
      testErr = np.zeros(201)
      for k in range(75, numRuns):
        trainData = data[:,0,k]
        fullTrainData = data[:,:,k]
        res = net.test(trainData, 201)
        err = net.meansqerr(fullTrainData,res,201)
        testErr = testErr + err
        #np.savetxt("results/data_hiddennodes" + runString + `k` + "_test_actual.csv", fullTrainData, delimiter=",", fmt="%d")
        if (k == 88):
          np.savetxt("results/data_hiddennodes" + runString + `k` + "_test_predicted.csv", res, delimiter=",", fmt="%d")
      testErr = testErr / (numRuns-75)
      print("Run " + `i` + " had " + `testErr[200]` + " test error.");
      np.savetxt("results/error_hiddennodes" + runString + `k` + "_test.csv", testErr, delimiter=",", fmt="%f")

      # Save the network thus far
      NetworkWriter.writeToFile(net, "results/network_hiddennodes" + `numHiddenNodes` + "_epochs" + `i` +  "_run" + `k` + ".xml")
      print("Run took %s s" % (time.time() - starttime))
  
main()      

