from keras.models import Sequential
from keras.layers.core import Activation, Dense

def createLstm(numHiddenUnits, numMolecules):
    # Make network
    model = Sequential()

    # Make hidden layer
    model.add(LSTM(output_dim=numHiddenUnits, input_dim=numMolecules, activation='sigmoid', return_sequences=True))
    model.add(LSTM(numHiddenUnits, activation='sigmoid', return_sequences=False))

    # Make output layer
    model.add(Dense(output_dim=numMolecules, input_dim=numHiddenUnits))
    model.add(Activation("sigmoid"))

    return model
