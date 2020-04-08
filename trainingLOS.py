from sklearn import svm
import csv
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# feature indices
ID = 0
SUBJECT_ID = 1
HADM_ID = 2
ADMISSION_TYPE = 3
ADMISSION_LOCATION = 4
DISCHARGE_LOCATION = 5
INSURANCE = 6
MARITAL_STATUS = 7
GENDER = 8
AGE = 9
Service_count = 10
icu_LOS = 11
LOS = 12

# training features
start = ADMISSION_TYPE
end = icu_LOS

testPercentage = 0.10

# [0, 0.5] -> class 0
# [0.5, 1.5] -> class 1
# ...
# [num, -] -> class x
#classOfLOS = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 30, 40, 50, 60, 70]
classOfLOS = [0, 5,10,15,28]
#classOfLOS = [0, 3, 5, 10, 20, 40]

with open(os.getcwd() + "/data/patientInfoFinal.csv", 'r') as file:
    line = csv.reader(file, delimiter='\t')
    train_data = list(line)

for i in range(1, len(train_data)):
    str = train_data[i][0];
    train_data[i] = str.split(',')
for i in range(1, len(train_data)):
    for j in range(0, len(train_data[i])):
        train_data[i][j] = float(train_data[i][j])

X = train_data[1: len(train_data)]  # jump header
y = [None]*len(X)
for i in range(0, len(X)):
    for j in range(0, len(X[i])):
        X[i][j] = float(X[i][j])

# classify LOS
classOfLOSNum = [0]*len(classOfLOS)
for i in range(0, len(X)):
    for j in range(0, len(classOfLOS)-1):
        if X[i][LOS]>=classOfLOS[j] and X[i][LOS]<=classOfLOS[j+1]:
            classOfLOSNum[j] += 1
            X[i][LOS] = j
            break
    if X[i][LOS] >= classOfLOS[len(classOfLOS)-1]:
        classOfLOSNum[len(classOfLOS)-1] += 1
        X[i][LOS] = len(classOfLOS)-1

for i in range(0, len(X)):
    y[i] = X[i][LOS]
    X[i] = X[i][start:end+1]

X_train = []
y_train = []
X_test = []
y_test_truelabel = []
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
        y_test_truelabel.append(y[k])
    start = end

'''
print(len(X))
print(len(X_train))
print(len(X_test))

clf = ExtraTreesClassifier(n_estimators=100)
clf.fit(X, y)
print(clf.feature_importances_)

'''

clf = svm.SVC()
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)

print("Number in class: ", classOfLOSNum)
correct = 0
for i in range(0, len(y_test)):
    if y_test[i] == y_test_truelabel[i]:
        correct += 1
    print("y_test: ",y_test[i],",y_test_true: ",+y_test_truelabel[i])
print("Overall accuracy: ", correct/len(y_test))

start = 0
end = 0
classCorrect = [0]*len(classOfLOSNum)
for i in range(0, len(testNum)):
    correct = 0
    end += testNum[i]
    for j in range(start, end):
        if y_test[j] == y_test_truelabel[j]:
            correct += 1
    classCorrect[i] = correct
    start = end
    print("class[", i, "] accuracy: ", correct/testNum[i])

correct = sum(classCorrect)

print("Overall accuracy: ", correct/len(y_test))
