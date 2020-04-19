def testSVM(prediction, y_test, testNum):
    start = 0
    end = 0
    classCorrect = [0] * len(testNum)
    for i in range(0, len(testNum)):
        correct = 0
        end += testNum[i]
        for j in range(start, end):
            #if i == 2:
            #   print("y_test: ", y_test[j], ", y_test_true: ", + y_test_truelabel[j])
            if prediction[j] == y_test[j]:
                correct += 1
        classCorrect[i] = correct
        start = end
        acc = 0
        if testNum[i] != 0:
            acc = correct / testNum[i]
        print("class[", i, "] accuracy: ", acc)
    correct = sum(classCorrect)
    print("Overall accuracy: ", correct / len(prediction))

def testSVR(prediction, y_test):
    MSE = 0
    R_square_denominator = 0
    y_bar = max(y_test) - min(y_test)
    for i in range(len(prediction)):
        #print("y_test: ", y_test[i], " y_test_truelabel: ", y_test_truelabel[i])
        MSE += (y_test[i] - prediction[i])**2
        R_square_denominator += (y_test[i] - y_bar)**2
    R_square_numerator = MSE
    MSE /= len(prediction)
    R_square = 1 - R_square_numerator/R_square_denominator
    print("MSE: ", MSE)
    print("R_square: ", R_square)

def test_simple(y_test, prediction):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == prediction[i]:
            correct += 1
    print("Test accuracy:", correct/len(y_test))

