from sklearn import svm
import csv
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

# feature indices
ID = 0
SUBJECT_ID = 1
ADMISSION_TYPE = 2
ADMISSION_LOCATION = 3
DISCHARGE_LOCATION = 4
INSURANCE = 5
MARITAL_STATUS = 6
GENDER = 7
AGE = 8
Service_count = 9
icu_LOS = 10
LOS = 11

# training features
start = ADMISSION_TYPE
end = icu_LOS

testPercentage = 0.1

# [0, 0.5] -> class 0
# [0.5, 1.5] -> class 1
# ...
# [num, -] -> class x
#classOfLOS = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 30, 40, 50, 60, 70]
classOfLOS = [0, 3, 5, 7, 10, 12, 15, 20, 28, 40]
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

for i in range(0, len(X)): 12,
    y[i] = X[i][LOS]
    X[i] = X[i][start:end+1]

 12,sampleLen = len(X)

X_train = []
y_train = []
X_test = []
y_test_truelabel = []

start = 0
end = 0
for j in range(0, len(classOfLOSNum)):
    end += classOfLOSNum[j]
    train_len = (end - start) * sampleLen / len(X) * (1 - testPercentage)
    test_len = (end - start) * sampleLen / len(X)
    for k in range(start, start + int(train_len)):
        X_train.append(X[k])
        y_train.append(y[k])
    for k in range(start + int(train_len), start + int(test_len)):
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
correct = 0
for i in range(0, len(y_test)):
    print("y_test: ",y_test[i],",y_test_true: ",+y_test_truelabel[i])
    if y_test[i] == y_test_truelabel[i]:
        correct += 1

print(correct/len(y_test))
