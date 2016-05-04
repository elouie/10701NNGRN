# Welcome to Genetic Regulatory Network predictive modeling!
# This code generates a neural network to run simulations with.
# These are the steps:

import numpy as np
from mlp import MLP
from load_data import data_load
from readArgs import readArgs
from createOrLoadModel import createOrLoadModel
from trainModel import trainModel
from dirmake import dirmake

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
    loadDataFname = args.loadDataFname
    saveDataFname = args.saveDataFname
    loadNetworkFname = args.loadNetworkFname
    saveNetworkFname = args.saveNetworkFname
    learningRate = args.learningRate
    learnerType = args.learnerType

    resultsDir = "results/{}_hl_{}_hn_{}_runs_{}_mol_{}_ts_{}_lr_{:0.4f}/".format(learnerType, numHiddenLayers, numHiddenUnits, numRuns, numMolecules, numTimesteps, learningRate)
    modelsDir = "models/{}_hl_{}_hn_{}_runs_{}_mol_{}_ts_{}_lr_{:0.4f}/".format(learnerType, numHiddenLayers, numHiddenUnits, numRuns, numMolecules, numTimesteps, learningRate)

    s = []
    s.append("Beginning {} training with parameters:".format(learnerType))
    s.append("\tNumber of hidden units: {}".format(numHiddenUnits))
    s.append("\tNumber of hidden layers: {}".format(numHiddenLayers))
    s.append("\tNumber of molecules: {}".format(numMolecules))
    s.append("\tNumber of timesteps: {}".format(numTimesteps))
    s.append("\tNumber of testing timesteps: {}".format(numTestTimesteps))
    s.append("\tNumber of runs per input test file: {}".format(numRuns))
    s.append("\tMaximum epochs to train over: {}".format(maxEpochs))
    s.append("\tFile name to load data: {}".format(loadDataFname))
    s.append("\tFile name to save data: {}".format(resultsDir))
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

    dirmake(resultsDir)
    dirmake(modelsDir)

    # Set all save/load names to specific directories:
    loadDataFname = "data/" + loadDataFname
    saveNetworkFname = "models/" + saveNetworkFname
    if not loadNetworkFname is None:
        loadNetworkFname = "models/" + loadNetworkFname

    # Set up the initial network
    model = createOrLoadModel(loadNetworkFname, learnerType, numHiddenUnits, numMolecules, learningRate)

    # Load the data into a matrix for use over epochs
    data = np.zeros((numRuns, numMolecules, numTimesteps), dtype=int)
    for i in range(numRuns):
        data[i,:,:] = data_load(loadDataFname,i,i+1, numMolecules, numTimesteps)

    # Train the data
    trainModel(model, data, maxEpochs, modelsDir, resultsDir)

main()



