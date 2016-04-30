def readOrCreateNetwork(loadFName, learnerType, numHiddenUnits, numMolecules, learningRate):
    model
    # Create a network:
    if learnerType == "lstm":
        print "LSTM is not currently supported."
    else:
        model = createMlp(numHiddenUnits, numMolecules, learningRate)
    if not loadFName == "":
        loadPath = "models/" + loadFName
        model = model_from_json(open(loadPath + '.json').read())
        model.load_weights(loadPath + '.h5')
