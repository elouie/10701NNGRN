from keras.models import model_from_json
from keras.utils.visualize_util import plot
from runTestsAndSave import runTestsAndSave
from math import floor
from time import time
import numpy as np

def trainModel(model, data, epochs, saveModelFname, saveDataFname):
    numRuns = data.shape[0]
    numMols = data.shape[1]
    numTimesteps = data.shape[2]
    numTrainRuns = int(floor(numRuns*0.75))
    numValidationRuns = numRuns - numTrainRuns

    # Split data into training data and test data:
    training_data = data[:numTrainRuns,:,:]
    train_data_x = training_data[:,:,0:-1]
    train_init_x = training_data[:,:,0]
    train_data_y = training_data[:,:,1:]
    validation_data = data[numTrainRuns:,:,:]
    validation_data_x = validation_data[:,:,0:-1]
    validation_init_x = validation_data[:,:,0]
    validation_data_y = validation_data[:,:,1:]

    # Filenames:
    gradTrainFile = saveDataFname + "grad_train_epoch_{}.csv"
    gradValidationFile = saveDataFname + "grad_validation_epoch_{}.csv"
    avgGradTrainFile = saveDataFname + "avg_grad_train_final_epoch_{}.csv"
    avgGradValidationFile = saveDataFname + "avg_grad_validation_final_epoch_{}.csv"
    avgTrainFile = saveDataFname + "avg_train_final_epoch_{}.csv"
    avgValidationFile = saveDataFname + "avg_validation_final_epoch_{}.csv"
    modelJsonFile = saveModelFname + 'model_run_{}.json'
    modelHdfFile = saveModelFname + 'model_run_{}.h5'

    print "Beginning training..."
    totalstarttime = time()

    train_err = np.zeros(numTrainRuns)
    test_err = np.zeros(numValidationRuns)

    threshold = 0.000001
    avg_grad_train_err = np.zeros(epochs+1)
    avg_grad_train_err[0] = 1
    avg_grad_test_err = np.zeros(epochs)
    avg_train_err = np.zeros(epochs)
    avg_test_err = np.zeros(epochs)

    # For each epoch
    for i in range(epochs):
        starttime = time()

        # For each run
        for j in range(numTrainRuns):
            train_err[j] = model.train_on_batch(train_data_x[j,:,:].T, train_data_y[j,:,:].T)
        np.savetxt(gradTrainFile.format(i), train_err, delimiter=",", fmt="%6f")
        for j in range(numValidationRuns):
            test_err[j] = model.test_on_batch(validation_data_x[j,:,:].T, validation_data_y[j,:,:].T)
        np.savetxt(gradValidationFile.format(i), test_err, delimiter=",", fmt="%6f")

        print "Train time for epoch {} was {:0.4f}".format(i, time() - starttime)

        # Run predictions on training data and output error and predictions to file
        avg_train_err[i] = runTestsAndSave(model, train_init_x, train_data_y, numTrainRuns, numMols, numTimesteps, saveDataFname + "train_epoch_" + `i`)
        avg_test_err[i] = runTestsAndSave(model, validation_init_x, validation_data_y, numValidationRuns, numMols, numTimesteps, saveDataFname + "validation_epoch_" + `i`)

        # Visualize and save model
        #modelFname=saveModelFname + '_run_' + `i`
        #plot(model, to_file=(modelFname + '.png'))

        json_string = model.to_json()
        open(modelJsonFile.format(i), 'w+').write(json_string)
        model.save_weights(modelHdfFile.format(i))

        print "Execution time for epoch {} was {:0.4f}".format(i, time() - starttime)
        avg_grad_train_err[i+1] = np.average(train_err)
        avg_grad_test_err[i] = np.average(test_err)
        if abs(avg_grad_train_err[i] - avg_grad_train_err[i+1]) < threshold:
            break

        print "Training completed with gradient train error {:0.6f}, test {:0.6f}, predicted train error {:0.6f}, test {:0.6f}.".format(avg_grad_train_err[i+1], avg_grad_test_err[i], avg_train_err[i], avg_test_err[i])

    np.savetxt(avgTrainFile.format(i), avg_train_err[0:i], delimiter=",", fmt="%6f")
    np.savetxt(avgValidationFile.format(i), avg_test_err[0:i], delimiter=",", fmt="%6f")
    np.savetxt(avgGradTrainFile.format(i), avg_grad_train_err[1:i+1], delimiter=",", fmt="%6f")
    np.savetxt(avgGradValidationFile.format(i), avg_grad_test_err[0:i], delimiter=",", fmt="%6f")

    print "Total execution time for {} epochs was {:0.4f}".format(i, time() - totalstarttime)
