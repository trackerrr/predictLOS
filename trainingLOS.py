import csv
import os
import trainingSVM
import trainingNN
import featureRanking
import pickle

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

test_percentage = 0.2

with open(os.getcwd() + "/data/patientInfoFinal.csv", 'r') as file:
    line = csv.reader(file, delimiter='\t')
    train_data = list(line)

for i in range(1, len(train_data)):
    str = train_data[i][0]
    train_data[i] = str.split(',')
for i in range(1, len(train_data)):
    for j in range(0, len(train_data[i])):
        train_data[i][j] = float(train_data[i][j])

X = train_data[1: len(train_data)] # jump header

lower_bound_LOS = 0
upper_bound_LOS = 50
index = 0
for i in range(0, len(X)):
    if X[index][LOS] > upper_bound_LOS or X[index][LOS] < lower_bound_LOS:
        X.pop(index)
    else:
        index += 1
print(lower_bound_LOS, "<= LOS <=", upper_bound_LOS)
print("Total size: ", len(X))

features_class = [AGE, icu_LOS]
features_regr = [ADMISSION_LOCATION, DISCHARGE_LOCATION, MARITAL_STATUS, AGE, Service_count, icu_LOS]
# classOfLOS = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 30, 40, 50, 60, 70]
#classOfLOS = [0, 5, 10, 15, 28]
classOfLOS = [0, 10, 30]
# classOfLOS = [0, 3, 5, 10, 20, 40]
# print("LOS classes: ", classOfLOS)
#trainingSVM.SVM(X, features, LOS, classOfLOS)
#trainingSVM.SVR(X, features_regr, LOS)
#pickle.dump(classifyModel, open("classifyModel", 'wb'))
#pickle.dump(regressionModel, open("regressionModel", 'wb'))

trainingSVM.combined_SVM_SVR(X, features_regr, features_regr, LOS, classOfLOS, test_percentage)
'''
ranking = featureRanking.rankFeatures(X, y)
print("feature importance: ", ranking)
'''