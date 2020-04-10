
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

def rankFeatures (features, target):
    clf = ExtraTreesClassifier(n_estimators=100)
    clf.fit(features, target)
    return clf.feature_importances
