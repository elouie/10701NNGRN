from keras.models import Sequential
from keras.layers.core import Activation, Dense

def createMlp(numHiddenUnits, numMolecules):
    # Make network
    model = Sequential()

    # Make hidden layer
    model.add(Dense(output_dim=numHiddenUnits, input_dim=numMolecules))
    model.add(Activation("sigmoid"))

    # Make output layer
    model.add(Dense(output_dim=numMolecules, input_dim=numHiddenUnits))
    model.add(Activation("sigmoid"))

    return model
