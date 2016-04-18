# runSimulation(network, testData, fname)
#   network: An MLP object
#   testData: Data to run a simulation on
#   fname: A file to save the results of the simulation.
#
# This file runs a simulation given data, then generates two results:
# the predicted values of the simulation over a number of steps and
# the mean squared error per time step difference of actual values,
# to be saved to "$RESULTS_DIR/$fname.csv" and "$ERR_DIR/$fname.csv",
# respectively.
def runSimulation(network, testData, fname):
    # TODO
