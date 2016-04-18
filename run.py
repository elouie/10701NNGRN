# Welcome to Genetic Regulatory Network predictive modeling!
# This code generates a neural network to run simulations with.
# These are the steps:

import numpy as np
import time
from mlp import MLP
from load_data import data_load

def main():
  # BEGIN load command line arguments
  # Options:
  #   loadNetworkFname: The file name of a previously saved network to load
  #   saveNetworkFname: The file name to save a network to
  #   numHiddenNodes: The number of hidden nodes to use in the hidden layer
  #   numHiddenLayers: The number of hidden layers to use before the output layer
  #   numMolecules: The number of rows/molecules in the data
  #   numTimesteps: The number of timesteps that the input runs on
  #   numTestTimesteps: The number of timesteps to generate during testing (Useful for evaluating longterm insights)
  #   numRuns: The number of runs in each initialized state dataset
  #   numEpochs: Maximum number of epochs to allow the neural network to run
  #              (May be needed if no good local boundary)
  numHiddenNodes = 100
  numHiddenLayers = 1
  numMolecules = 124
  numTimesteps = 201
  numTestTimesteps = 201
  numRuns = 99
  numEpochs = 801
  # END load command line arguments

  # Set up the initial network
  net = MLP(numMolecules, numHiddenNodes, numHiddenLayers)

  # Load the data into a matrix for use over epochs
  data = np.zeros((numMolecules,numTimesteps,numRuns))
  for i in range(numRuns):
    data[:,:,i] = data_load('13A-no_0',i,i+1).astype(int)

  # Load the data into a matrix for use over epochs
  secondary_data = np.zeros((numMolecules,numTimesteps,numRuns))
  for i in range(numRuns):
    secondary_data[:,:,i] = data_load('14A-no_0',i,i+1).astype(int)

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

      # Get test error and output
      altTestErr = np.zeros(201)
      for k in range(75, numRuns):
        trainData = secondary_data[:,0,k]
        fullTrainData = secondary_data[:,:,k]
        res = net.test(trainData, 201)
        err = net.meansqerr(fullTrainData,res,201)
        altTestErr = altTestErr + err
        #np.savetxt("results/data_hiddennodes" + runString + `k` + "_test_actual.csv", fullTrainData, delimiter=",", fmt="%d")
        if (k == 88):
          np.savetxt("results/data_hiddennodes" + runString + `k` + "_alttest_predicted.csv", res, delimiter=",", fmt="%d")
      altTestErr = altTestErr / (numRuns-75)
      print("Run " + `i` + " had " + `altTestErr[200]` + " test error.");
      np.savetxt("results/error_hiddennodes" + runString + `k` + "_alttest.csv", altTestErr, delimiter=",", fmt="%f")

      # Save the network thus far
      #NetworkWriter.writeToFile(net, "results/network_hiddennodes" + `numHiddenNodes` + "_epochs" + `i` +  "_run" + `k` + ".xml")
    print("Run " + `i` + " took %s s" % (time.time() - starttime))

main()

