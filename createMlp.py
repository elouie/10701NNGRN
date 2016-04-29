from keras.models import Sequential

def createMlp(numHiddenUnits, numMolecules, learningRate=0.1, momentum=0.9):
    # Make network
    model = Sequential()

    # Make hidden layer
    model.add(Dense(output_dim=numHiddenUnits, input_dim=numMolecules))
    model.add(Activation("sigmoid"))

    # Make output layer
    model.add(Dense(output_dim=numMolecules, input_dim=numHiddenUnits))
    model.add(Activation("sigmoid"))

    # Define training function
    model.compile(loss='mean_squared_error', optimizer=SGD(lr=learningRate, momentum=momentum, nesterov=True))

    return model
