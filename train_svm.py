from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from preprocess import processNumData, processCatData, divideDataset, get_X_y
import warnings
warnings.filterwarnings("ignore")

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

clf = SVC(kernel='rbf', C=10, gamma='auto')
clf.fit(processed_train_data, y)

pred = clf.predict(processed_test_data)
pred = pd.DataFrame(pred)
pred.columns = ['Survived']

submission = pd.concat([test['PassengerId'], pred], axis=1)
submission.to_csv('svm_normalized.csv', index=False)
print 'End'
