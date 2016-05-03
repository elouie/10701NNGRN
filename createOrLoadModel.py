from createMlp import createMlp
from keras.models import model_from_json
from keras.optimizers import SGD

def createOrLoadModel(loadFName, learnerType, numHiddenUnits, numMolecules, learningRate=0.1, momentum=0.9):
    # Create a network:
    if learnerType == "lstm":
        print "LSTM is not currently supported."
    else:
        model = createMlp(numHiddenUnits, numMolecules)

    # Load previous network if supplied
    if not loadFName is None:
        loadPath = "models/" + loadFName
        model = model_from_json(open(loadPath + '.json').read())
        model.load_weights(loadPath + '.h5')

    model.compile(loss='mean_squared_error', optimizer=SGD(lr=learningRate, momentum=momentum, nesterov=True))
    return model
