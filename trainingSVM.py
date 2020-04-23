import test
import numpy as np
import plot
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pickle
from random import sample

def SVM (X, features, LOS_pos, classOfLOS):
    y = [None] * len(X)
    testPercentage = 0

    # plot.plot_column(np.array(X), icu_LOS, LOS, "icu_LOS", "LOS")

    # classify target
    classOfLOSNum = [0] * len(classOfLOS)
    for i in range(0, len(X)):
        for j in range(0, len(classOfLOS) - 1):
            if X[i][LOS_pos] >= classOfLOS[j] and X[i][LOS_pos] <= classOfLOS[j + 1]:
                classOfLOSNum[j] += 1
                X[i][LOS_pos] = j
                break
        if X[i][LOS_pos] >= classOfLOS[len(classOfLOS) - 1]:
            classOfLOSNum[len(classOfLOS) - 1] += 1
            X[i][LOS_pos] = len(classOfLOS) - 1
    print("classOfLOSNum: ", classOfLOSNum)

    # plot.plot_column(np.array(X), icu_LOS, LOS, "icu_LOS", "LOS classes")

    # select features
    for i in range(0, len(X)):
        arr = [0] * len(features)
        y[i] = X[i][LOS_pos]
        for j in range(0, len(features)):
            arr[j] = X[i][features[j]]
        X[i] = arr

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    testNum = []
    start = 0
    end = 0
    for j in range(0, len(classOfLOSNum)):
        end += classOfLOSNum[j]
        train_len = (end - start) * (1 - testPercentage)
        testNum.append(end - int(train_len) - start)
        for k in range(start, start + int(train_len)):
            X_train.append(X[k])
            y_train.append(y[k])
        for k in range(start + int(train_len), end):
            X_test.append(X[k])
            y_test.append(y[k])
        start = end

    print("===Test - SVM")
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
# test
    test.testSVM(prediction, y_test, testNum)
# plot
    plot.plot_map(y_test, prediction, "y_test_truelabel", "prediction", "SVM final result")
    return clf

def SVR (X, features, LOS_pos):
    print("===Test - SVR")
    y = [None] * len(X)
    for i in range(0, len(X)):
        arr = [0] * len(features)
        y[i] = X[i][LOS_pos]
        for j in range(0, len(features)):
            arr[j] = X[i][features[j]]
        X[i] = arr
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cross_validation(X_train, y_train, 3)
    # final model
    clf = svm.SVR()
    clf.fit(X, y)
    prediction = clf.predict(X_test)
    plot.plot_map(y_test, prediction, "y_test", "prediction", "SVR final result")
    return clf

def combined_SVM_SVR (X, features_class, features_regr, LOS_pos, classOfLOS, test_percentage):
    print("===Test - Combined SVM SVR")
    X_features_class = [None] * len(X)
    X_features_regr = [None] * len(X)
    y = [None] * len(X)
    for i in range(0, len(X)):
        arr_class = [0] * len(features_class)
        arr_regr = [0] * len(features_regr)
        y[i] = X[i][LOS_pos]
        for j in range(len(features_class)):
            arr_class[j] = X[i][features_class[j]]
            arr_regr[j] = X[i][features_regr[j]]
        X_features_class[i] = arr_class
        X_features_regr[i] = arr_regr

    X_features_class = np.array(X_features_class)
    X_features_regr = np.array(X_features_regr)
    y = np.array(y)
    y_class = np.copy(y)
    classOfLOSNum = [0]*len(classOfLOS)
    for i in range(len(y)):
        for j in range(len(classOfLOS) - 1):
            if y[i] >= classOfLOS[j] and y[i] <= classOfLOS[j + 1]:
                y_class[i] = j
                classOfLOSNum[j] += 1
                break
        if y[i] >= classOfLOS[len(classOfLOS) - 1]:
            y_class[i] = len(classOfLOS) - 1
            classOfLOSNum[len(classOfLOS) - 1] += 1
    print("classOfLOSNum =", classOfLOSNum)
    X_train_class, X_test_class, X_train_regr, X_test_regr, y_train, y_test, y_train_class, y_test_class = train_test_split(X_features_class, X_features_regr, y, y_class, test_size=test_percentage)

    print("===SVM")
    clf = svm.SVC()
    clf.fit(X_train_class, y_train_class)
    class_prediction = clf.predict(X_test_class)
    test.test_simple(y_test_class, class_prediction, len(classOfLOS))
    pickle.dump(clf, open("model/svm_model", 'wb'))

    print("===SVR")
    prediction = np.empty((0))
    y_test_rearrange = np.empty((0))
    for i in range(len(classOfLOS)-1):
        print("SVR training set", i)
        indices = []
        for j in range(len(y_train_class)):
            if y_train_class[j] == i:
                indices.append(j)
        X_train_sub = np.array(X_train_regr[indices])
        y_train_sub = np.array(y_train[indices])
        indices = []
        for j in range(len(y_test_class)):
            if class_prediction[j] == i:
                indices.append(j)
        X_test_sub = np.array(X_test_regr[indices])
        y_test_sub = np.array(y_test[indices])
        clf = svm.SVR()
        clf.fit(X_train_sub, y_train_sub)
        fname = "model/svr_model_" + str(i)
        pickle.dump(clf, open(fname, 'wb'))
        cross_validation(X_train_sub, y_train_sub, 3)
        regression_pred = clf.predict(X_test_sub)
        prediction = np.append(prediction, regression_pred)
        y_test_rearrange = np.append(y_test_rearrange, y_test_sub)
        test.testSVR(regression_pred, y_test_sub)
        plot.plot_map(y_test_sub, regression_pred, "y_test", "prediction", "regression subset")

    print("Overall:")
    indices = []
    for i in range(len(y_test_rearrange)):
        if y_test_rearrange[i] <= 30:
            indices.append(i)
    prediction = np.array(prediction[indices])
    y_test_rearrange = np.array(y_test_rearrange[indices])
    test.testSVR(prediction, y_test_rearrange)
    plot.plot_map(y_test_rearrange, prediction, "y_test", "prediction", "final regression result")

    interval = 5
    for i in range(classOfLOS[0], classOfLOS[len(classOfLOS) - 1], interval):
        indices = []
        for j in range(len(prediction)):
            if y_test_rearrange[j] >= i and y_test_rearrange[j] <= i + interval:
                indices.append(j)
        y_test_rearrange_sub = np.array(y_test_rearrange[indices])
        prediction_sub = np.array(prediction[indices])
        print("class", i, "-", i + interval)
        test.testSVR(prediction_sub, y_test_rearrange_sub)



def cross_validation(X, y, kfold):
    kf = KFold(n_splits=kfold, shuffle=True)
    k = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("k =", k)
        k += 1
        clf = svm.SVR()
        clf.fit(X_train, y_train)
        prediction = clf.predict(X_test)
        # test
        test.testSVR(prediction, y_test)
        # plot
        plot.plot_map(y_test, prediction, "y_test", "prediction", "cross validation")