from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from runTestsAndSave import runTestsAndSave
from math import floor

def trainModel(model, data, epochs, saveModelFname, saveDataFname):
  numRuns = data.shape[0]
  numMols = data.shape[1]
  numTimesteps = data.shape[2]
  numTrainRuns = math.floor(numRuns*0.75)
  numValidationRuns = numRuns - numTrainRuns

  # Split data into training data and test data:
  training_data = data[numTrainRuns:,:,:]
  train_data_x = training_data[:,:,0:-1]
  train_data_y = training_data[:,:,1:]
  validation_data = data[numTrainRuns+1:,:,:]
  validation_data_x = validation_data[:,:,0:-1]
  validation_data_y =  validation_data[:,:,1:]

  # For each epoch
  for i in range(numEpochs):
    starttime = time.time()

    # For each run
    for j in range(75):
        model.train_on_batch(train_data_x[i,:,:], Y_batch[i,:,:])

    # Run predictions on training data and output error and predictions to file
    runTestsAndSave(model, train_data_x, train_data_y, saveDataFname + "_run_" + `i` + "_train")
    runTestsAndSave(model, train_data_x, train_data_y, saveDataFname + "_run_" + `i` + "_validation")

    # Visualize and save model
    modelFname='models/' + saveModelFname + '_run_' + `i`
    plot(model, to_file=(modelFname + '.png'))

    json_string = model.to_json()
    open(modelFname + '.json', 'w').write(json_string)
    model.save_weights(modelFname + '.h5')
