import csv
import os
import trainingSVM
import trainingNN
import featureRanking
import test
import matplotlib.pyplot as plt
import numpy as np

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
features = [ADMISSION_TYPE, INSURANCE, AGE, icu_LOS]
start = AGE
end = icu_LOS

testPercentage = 0.2

# [0, 0.5] -> class 0
# [0.5, 1.5] -> class 1
# ...
# [num, -] -> class x
#classOfLOS = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 30, 40, 50, 60, 70]
classOfLOS = [0, 5,10,15,28]
#classOfLOS = [0, 5,10]
#classOfLOS = [0, 3, 5, 10, 20, 40]
print("LOS classes: ", classOfLOS)

with open(os.getcwd() + "/data/patientInfoFinal.csv", 'r') as file:
    line = csv.reader(file, delimiter='\t')
    train_data = list(line)

for i in range(1, len(train_data)):
    str = train_data[i][0]
    train_data[i] = str.split(',')
for i in range(1, len(train_data)):
    for j in range(0, len(train_data[i])):
        train_data[i][j] = float(train_data[i][j])

X = train_data[1: len(train_data)]  # jump header
y = [None] * len(X)
for i in range(0, len(X)):
    for j in range(0, len(X[i])):
        X[i][j] = float(X[i][j])

def plot(X, x_pos, y_pos, x_label, y_label):
    plt.scatter(X[:, x_pos],
                X[:, y_pos],
                cmap='viridis')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
plot(np.array(X), icu_LOS, LOS, "icu_LOS", "LOS")

# classify target
classOfLOSNum = [0] * len(classOfLOS)
for i in range(0, len(X)):
    for j in range(0, len(classOfLOS) - 1):
        if X[i][LOS] >= classOfLOS[j] and X[i][LOS] <= classOfLOS[j + 1]:
            classOfLOSNum[j] += 1
            X[i][LOS] = j
            break
    if X[i][LOS] >= classOfLOS[len(classOfLOS) - 1]:
        classOfLOSNum[len(classOfLOS) - 1] += 1
        X[i][LOS] = len(classOfLOS) - 1
print("classOfLOSNum: ", classOfLOSNum)

plot(np.array(X), icu_LOS, LOS, "icu_LOS", "LOS classes")

for i in range(0, len(X)):
    y[i] = X[i][LOS]
    arr = [0] * len(features)
    for j in range(0, len(features)):
        arr[j] = X[i][features[j]]
    X[i] = arr

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
ranking = featureRanking.rankFeatures(X, y)
print("feature importance: ", ranking)

y_test = trainingSVM.trainSVM(X_train, y_train, X_test)
test.testModel(y_test, y_test_truelabel, testNum)
'''
trainingNN.NN(X_train, y_train, X_test, y_test_truelabel, classOfLOS)