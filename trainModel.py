from keras.optimizers import SGD

def trainModel(model, X_train, Y_train, epochs, learningRate, momentum):
    model.train_on_batch(X_batch, Y_batch)
