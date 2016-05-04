import argparse

# args = readArgs()
#
# Options:
#   loadNetworkFname: The file name of a previously saved network to load
#   saveNetworkFname: The file name to save a network to
#   numHiddenUnits: The number of hidden nodes/units to use in the hidden layer
#   numHiddenLayers: The number of hidden layers to use before the output layer
#   numMolecules: The number of rows/molecules in the data
#   numTimesteps: The number of timesteps that the input runs on
#   numTestTimesteps: The number of timesteps to generate during testing (Useful for evaluating longterm insights)
#   numRuns: The number of runs in each initialized state dataset
#   numEpochs: Maximum number of epochs to allow the neural network to run
#              (May be needed if no good local boundary)
def readArgs():
    parser = argparse.ArgumentParser(description="This program trains a ")
    parser.add_argument("-u", "--numHiddenUnits", default=100, type=int, help="The number of hidden nodes/units to use in the hidden layer")
    parser.add_argument("-l", "--numHiddenLayers", default=1, type=int, help="The number of hidden layers to use before the output layer")
    parser.add_argument("-m", "--numMolecules", default=51, type=int, help="The number of rows/molecules in the data")
    parser.add_argument("-s", "--numTimesteps", default=401, type=int, help="The number of timesteps that the input runs on")
    parser.add_argument("-t", "--numTestTimesteps", default=401, type=int, help="The number of timesteps to generate during testing (Useful for evaluating longterm insights)")
    parser.add_argument("-r", "--numRuns", default=200, type=int, help="The number of runs in each initialized state dataset")
    parser.add_argument("-e", "--maxEpochs", default=1000, type=int, help="Maximum number of epochs to allow the neural network to run (May be needed if no good local boundary)")
    parser.add_argument("-d", "--loadDataFname", default="", help="The file name where data should be loaded from")
    parser.add_argument("-x", "--saveDataFname", default="", help="The file name where results should be saved to")
    parser.add_argument("-i", "--loadNetworkFname", default=None, help="The file name of a previously saved network to load")
    parser.add_argument("-o", "--saveNetworkFname", default="", help="The file name to save a network to")
    parser.add_argument("-p", "--learningRate", default=0.02, type=float, help="The rate at which the network will learn")
    parser.add_argument("-f", "--learnerType", default="MLP")

    return parser.parse_args()
