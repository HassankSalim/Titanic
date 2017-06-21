from sklearn.preprocessing import OneHotEncoder, Normalizer, LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold


ohe = OneHotEncoder()
lbl = LabelEncoder()
normalizer = Normalizer()
min_max_scaler = MinMaxScaler()

df = pd.read_csv('data/train.csv')

feature_list = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
target = 'Survived'
categorical_features = ['Pclass', 'Sex', 'Embarked']
numical_features = ['Age', 'SibSp', 'Fare']
eval_size = 10

X = df[feature_list]
y = df[target]

kf = StratifiedKFold(y, eval_size)
train_index, test_index = next(iter(kf))

X_train, X_test = X.iloc[train_index], X.iloc[test_index]
y_train, y_test = y.iloc[train_index], y.iloc[test_index]
main_frame = X


age_means = main_frame.groupby('Sex').mean().Age
def fill_age(gender, age):
    if np.isnan(age):
        return age_means[gender]
    return age

fare_means = main_frame.groupby('Pclass').mean().Fare

def get_X_y():
    return df[feature_list], df[target]

def fill_fare(pclass, fare):
    if np.isnan(fare):
        return fare_means[pclass]
    return fare

def specialProcessing(frame, main_frame):
    frame.Age.fillna(main_frame.Age.mean(), inplace = True)
    frame.Embarked.fillna(main_frame.Embarked.value_counts().index[0], inplace = True)
    frame.Fare = frame.apply(lambda x : fill_fare(x['Pclass'], x['Fare']), axis = 1)
    frame.fillna(main_frame.median())
    return frame

specialProcessing(X, main_frame)

def divideDataset():
    return X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

def encodeCatData(frame, feature_name):
    lbl.fit(main_frame[feature_name])
    temp_for_ohe = lbl.transform(main_frame[feature_name]).reshape(-1, 1)
    temp = lbl.transform(frame[feature_name]).reshape(-1, 1)
    ohe.fit(temp_for_ohe)
    temp = ohe.transform(temp).toarray()
    return temp

def processCatData(frame):
    specialProcessing(frame, main_frame)
    out = []
    for i in categorical_features:
        out.append(encodeCatData(frame, i))
    output = np.hstack(out)
    return output

def encodeNumData(frame, feature_name):
    min_max_scaler.fit(main_frame[feature_name])
    temp = min_max_scaler.transform(frame[feature_name]).reshape(-1, 1)
    return temp

def processNumData(frame):
    specialProcessing(frame, main_frame)
    out = []
    for i in numical_features:
        out.append(encodeNumData(frame, i))
    output = np.hstack(out)
    return output
