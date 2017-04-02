from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from preprocess import processNumData, processCatData, divideDataset, get_X_y
from time import time
import warnings
from sys import argv
from pickle import dump
warnings.filterwarnings("ignore")


if len(argv) < 2:
    print('Usage : python train_svm.py file_name.pickle')
    exit()
else:
    pickle_file_name = argv[1]

def replace_zero_neg_one(val):
    if val:
        return 1
    return -1

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X, y = get_X_y()

# X_train, X_test, y_train, y_test = divideDataset()

# y_train.apply(lambda x : replace_zero_neg_one(x))
# y_test.apply(lambda x : replace_zero_neg_one(x))
y.apply(lambda x : replace_zero_neg_one(x))

processed_train_data = np.hstack((processCatData(X), processNumData(X)))
processed_test_data = np.hstack((processCatData(test), processNumData(test)))

params = { 'kernel' : ('linear', 'rbf'), 'C' : [1, 10, 100, 1000, 10000] }
clf = SVC(kernel='rbf', C=10, gamma='auto') #svc

# t = time()
# print('Start.....')
# clf = GridSearchCV(estimator = svc, param_grid = params, cv = 10, n_jobs=-1)
clf.fit(processed_train_data, y)
with open(pickle_file_name, 'wb') as f:
    dump(clf, f)
# print clf
# print('End.......')
# print('Time Elasped : ', time() - t)

# pred = clf.predict(processed_test_data)
# pred = pd.DataFrame(pred)
# pred.columns = ['Survived']
#
# submission = pd.concat([test['PassengerId'], pred], axis=1)
# submission.to_csv('grid_search_cv_svm_normalized.csv', index=False)
