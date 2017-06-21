import pandas as pd
import numpy as np
from preprocess import processNumData, processCatData, divideDataset, get_X_y
from time import time
import warnings
from sys import argv
from pickle import load
warnings.filterwarnings("ignore")

test = pd.read_csv('data/test.csv')
pickle_file_name = argv[1]
output_csv_name = argv[2]

processed_test_data = np.hstack((processCatData(test), processNumData(test)))

pickle_file = open(pickle_file_name, 'rb')
clf = load(pickle_file)

pred = clf.predict(processed_test_data)
print(pred)
pred = pd.DataFrame(pred)
pred.columns = ['Survived']

submission = pd.concat([test['PassengerId'], pred], axis=1)
submission.to_csv(output_csv_name, index=False)
