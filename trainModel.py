from keras.models import model_from_json
from keras.utils.visualize_util import plot
from runTestsAndSave import runTestsAndSave
from math import floor
from time import time

def trainModel(model, data, epochs, saveModelFname, saveDataFname):
    numRuns = data.shape[0]
    numMols = data.shape[1]
    numTimesteps = data.shape[2]
    numTrainRuns = int(floor(numRuns*0.75))
    print "numTrainRuns {}".format(numTrainRuns)
    numValidationRuns = numRuns - numTrainRuns

    # Split data into training data and test data:
    training_data = data[:numTrainRuns,:,:]
    train_data_x = training_data[:,:,0:-1]
    train_init_x = training_data[:,:,0]
    train_data_y = training_data[:,:,1:]
    validation_data = data[numTrainRuns:,:,:]
    validation_data_x = validation_data[:,:,0:-1]
    validation_init_x = validation_data[:,:,0]
    validation_data_y =    validation_data[:,:,1:]

    print "Beginning training..."
    totalstarttime = time()

    # For each epoch
    for i in range(epochs):
        starttime = time()

        # For each run
        for j in range(numTrainRuns):
            err = model.train_on_batch(train_data_x[j,:,:].T, train_data_y[j,:,:].T)
            print err

        print "Train time for epoch {} was {:10.4f}".format(j, time() - starttime)

        # Run predictions on training data and output error and predictions to file
        runTestsAndSave(model, train_init_x, train_data_y, numTrainRuns, numMols, numTimesteps, saveDataFname + "_run_" + `i` + "_train")
        runTestsAndSave(model, validation_init_x, validation_data_y, numValidationRuns, numMols, numTimesteps, saveDataFname + "_run_" + `i` + "_validation")

        # Visualize and save model
        #modelFname=saveModelFname + '_run_' + `i`
        #plot(model, to_file=(modelFname + '.png'))

        #json_string = model.to_json()
        #open(saveModelFname + '.json', 'w').write(json_string)
        #model.save_weights(saveModelFname + '.h5')

        print "Execution time for epoch {} was {:10.4f}".format(j, time() - starttime)
    print "Total execution time for {} epochs was {:10.4f}".format(epochs, time() - totalstarttime)
