import numpy as np
from meanSqErr import meanSqErr
from keras.models import Sequential

# runTestsAndSave(model, X, Y, ...)
#   model: A keras model
#   X: An n x m numpy array of n runs by m initial timesteps
#   Y: An n x m x p numpy array over p timesteps
def runTestsAndSave(model, X, Y, numRuns, numMols, numTimesteps, saveFName):
    savePath = saveFName + "_error_run_{}.csv"

    predictions = np.zeros((numRuns, numMols, numTimesteps))
    predictions[:,:,0] = X
    error = np.zeros(numTimesteps)

    # Predict next prediction based on previous prediction
    for k in range(1, numTimesteps):
        temp = model.predict(predictions[:,:,k-1], batch_size=numRuns)
        temp[temp < 0.5] = 0
        temp[temp >= 0.5] = 1
        predictions[:,:,k] = temp

    # Calculate error
    error = meanSqErr(Y, np.delete(predictions,0,2), numRuns, numMols, numTimesteps-1)
    #print("We have  " + `error[numTimesteps-2]` + " error.");

    # Save error output
    np.savetxt(savePath.format(k), error, delimiter=",", fmt="%6f")

    return np.average(error)
