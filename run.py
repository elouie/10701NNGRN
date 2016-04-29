# Welcome to Genetic Regulatory Network predictive modeling!
# This code generates a neural network to run simulations with.
# These are the steps:

import numpy as np
import time
from mlp import MLP
from load_data import data_load
from readArgs import readArgs

def main():
  # Load commandline arguments
  args = readArgs()
  numHiddenUnits = args.numHiddenUnits
  numHiddenLayers = args.numHiddenLayers
  numMolecules = args.numMolecules
  numTimesteps = args.numTimesteps
  numTestTimesteps = args.numTestTimesteps
  numRuns = args.numRuns
  maxEpochs = args.maxEpochs
  loadNetworkFname = args.loadNetworkFname
  saveNetworkFname = args.saveNetworkFname
  learningRate = args.learningRate
  learnerType = args.learnerType

  s = []
  s.append("Beginning {} training with parameters:".format(learnerType))
  s.append("\tNumber of hidden units: {}".format(numHiddenUnits))
  s.append("\tNumber of hidden layers: {}".format(numHiddenLayers))
  s.append("\tNumber of molecules: {}".format(numMolecules))
  s.append("\tNumber of timesteps: {}".format(numTimesteps))
  s.append("\tNumber of testing timesteps: {}".format(numTestTimesteps))
  s.append("\tNumber of runs per input test file: {}".format(numRuns))
  s.append("\tMaximum epochs to train over: {}".format(maxEpochs))
  s.append("\tFile name to load network from (Empty to not load): {}".format(loadNetworkFname))
  s.append("\tFile name to save network to: {}".format(saveNetworkFname))
  s.append("\tLearning rate of training: {}".format(learningRate))
  s.append("\tType of learner to use (MLP + LSTM): {}".format(learnerType))
  s.append("\nIs this good? (Y/N)")
  print "\n".join(s)

  answer = raw_input().lower()
  if not (answer == "y" or answer == "yes"):
      print "Stopping!"
      exit()

  # Set up the initial network
  model = createOrLoadModel(loadNetworkFname, learnerType, numHiddenUnits, numMolecules, learningRate)

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

