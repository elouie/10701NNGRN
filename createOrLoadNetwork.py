def readOrCreateNetwork(loadFName, learnerType, numHiddenUnits, numMolecules, learningRate):
    model
    # Create a network:
    if learnerType == "lstm":
        print "LSTM is not currently supported."
    else:
        model = createMlp(numHiddenUnits, numMolecules, learningRate)
    if not loadFName == "":
        loadPath = "./" + loadFName
        model.load_weights(loadPath)
