def runTestsAndSave():
    runString = `numHiddenNodes` + "_epochs" + `i` +  "_run"
    # Get train error and output
    error = np.zeros(201)
    for k in range(1, 75):
        testData = data[k,:,0]
        fullTestData = data[k,:,:]
        res = net.test(testData, 201)
        err = net.meansqerr(fullTestData,res,201)
        trainErr = trainErr + err
        if (k == 71):
            np.savetxt("results/data_hiddennodes" + runString + `k` + "_train_predicted.csv", res, delimiter=",", fmt="%d")
    trainErr = trainErr / 75
    print("Run " + `i` + " had " + `trainErr[200]` + " training error.");
    np.savetxt("results/error_hiddennodes" + runString + `k` + "_train.csv", trainErr, delimiter=",", fmt="%f")
