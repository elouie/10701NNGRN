def createOrLoadModel(loadFName, learnerType, numHiddenUnits, numMolecules, learningRate):
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
        model.compile(loss='mean_squared_error', optimizer=SGD(lr=learningRate, momentum=momentum, nesterov=True))
