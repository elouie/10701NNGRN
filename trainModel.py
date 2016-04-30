from keras.optimizers import SGD
from keras.utils.visualize_util import plot
from runTestsAndSave import runTestsAndSave

def trainModel(model, data, epochs):

  # Split data into training data and test data:
  train_data_x = data.delete(...)
  train_data_y = ...
  validation_data_x = ...
  validation_data_y = ...

  # For each epoch
  for i in range(numEpochs):
    starttime = time.time()

    # For each run
    for j in range(75):
        model.train_on_batch(train_data_x[i,:,:], Y_batch[i,:,:])

    # Run predictions on training data and output error and predictions to file
    runTestsAndSave(model, train_data_x, train_data_y, saveDataFname + `i`)

    # Visualize and save model
    modelFname='models/' + saveModelFname + '_run_' + `i`
    plot(model, to_file=(modelFname + '.png'))

    json_string = model.to_json()
    open(modelFname + '.json', 'w').write(json_string)
    model.save_weights(modelFname + '.h5')
