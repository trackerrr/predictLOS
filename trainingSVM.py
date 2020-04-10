from sklearn import svm

def trainSVM (X_train, y_train, X_test):
    print("training with SVM...")
    print("===Test - SVM")
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_test = clf.predict(X_test)
    return y_test

