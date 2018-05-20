import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn import tree

df = pd.read_csv('train.csv')
df.fillna(0, inplace=True)

def numeric_data(df):
    numeric_dict = {}

    for column in df.columns.values:
        def int_value(value):
            return numeric_dict[value]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            to_list = df[column].values.tolist()
            to_set = set(to_list)
            counter = 0
            for i in to_set:
                if i not in numeric_dict:
                    numeric_dict[i] = counter
                    counter +=1
            df[column] = list(map(int_value, df[column]))
    return df

df = numeric_data(df)

X = np.array(df.drop(['Survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['Survived'])

clf = svm.SVC()
clf.fit(X, y)

correct = 0
for i in range(len(X)):
    to_float = np.array(X[i].astype(float))
    reshaping = to_float.reshape(-1, len(to_float))
    prediction = clf.predict(reshaping)
    if prediction[0] == y[i]:
        correct += 1

accuracy = float(correct)/len(X)
print accuracy

test_file = pd.read_csv('test.csv')
test = pd.read_csv('test.csv')
test.fillna(0, inplace=True)
test = numeric_data(test)
test = np.array(test).astype(float)
test = preprocessing.scale(test)

correct = 0
for i in range(len(test)):
    to_float = np.array(test[i].astype(float))
    reshaping = to_float.reshape(-1, len(to_float))
    prediction = clf.predict(reshaping)
    print prediction
