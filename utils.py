import pandas as pd
import numpy as np
from sklearn.utils import shuffle


def data_preprocessing(filename):
    data = pd.read_csv(filename)
    data = data.fillna(0.0)
    data = shuffle(data)
    data = data.reset_index()

    X = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'index'], axis=1)

    sex = X['Sex'] == 'male'
    sex = pd.DataFrame(sex, dtype=np.float32)
    X['Sex'] = sex

    embarked = np.empty(len(X))

    for index, row in X.iterrows():
        if row['Embarked'] == 'S':
            embarked[index] = 1.0
        elif row['Embarked'] == 'C':
            embarked[index] = 2.0
        elif row['Embarked'] == 'Q':
            embarked[index] = 3.0

    X['Embarked'] = embarked
    return X


def data_split(X,k):
    x_train = X[0:int(k * len(X))]
    y_train = np.empty([int(k * len(X)), 2])
    x_test = X[int(k * len(X)):len(X)]
    y_test = np.empty([int((1-k) * len(X) + 1), 2])
    return x_train, y_train, x_test, y_test
