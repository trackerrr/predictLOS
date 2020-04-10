def testModel(y_test, y_test_truelabel, testNum):
    correct = 0
    for i in range(0, len(y_test)):
        if y_test[i] == y_test_truelabel[i]:
            correct += 1
        # print("y_test: ", y_test[i], ",y_test_true: ", +y_test_truelabel[i])
    # print("Overall accuracy: ", correct / len(y_test))

    start = 0
    end = 0
    classCorrect = [0] * len(testNum)
    for i in range(0, len(testNum)):
        correct = 0
        end += testNum[i]
        for j in range(start, end):
            if y_test[j] == y_test_truelabel[j]:
                correct += 1
        classCorrect[i] = correct
        start = end
        print("class[", i, "] accuracy: ", correct / testNum[i])

    correct = sum(classCorrect)

    print("Overall accuracy: ", correct / len(y_test))