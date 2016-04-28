from keras.models import Sequential

def createMlp(numHiddenUnits, numMolecules):
    # Make network
    model = Sequential()

    # Make hidden layer
    model.add(Dense(output_dim=numHiddenUnits, input_dim=numMolecules))
    model.add(Activation("relu"))

    # Make output layer
    model.add(Dense(output_dim=numMolecules, input_dim=numHiddenUnits))
    model.add(Activation("relu"))

    # Define training function
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learningRate, momentum=momentum, nesterov=True))

    return model
