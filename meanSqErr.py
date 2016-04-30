import numpy as np

# Get the mean squared error
def meanSqErr(input, res, numRuns, numMols, ts):
    error = np.zeros(ts)
    for i in range(ts):
        for j in numRuns:
            error[i] = nr.sum(np.square(input[j, :, i] - res[j, :, i]))/numMols
    return error/numRuns
