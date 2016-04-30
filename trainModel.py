from keras.optimizers import SGD

def trainModel(model, data, epochs, learningRate, momentum):

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

    runTestsAndSave(model, train_data_x, train_data_y)
