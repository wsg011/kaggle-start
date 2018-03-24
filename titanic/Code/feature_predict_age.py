# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def load_data():
    df = pd.read_csv('../input/train.csv')

    train = df[df["Age"].notnull()]
    test = df[df["Age"].isnull()]
    print(train.shape, test.shape)

    x_train = train.drop(["PassengerId", "Age"], axis=1)
    y_train = train["Age"].values
    print(x_train.columns)

    return x_train, y_train, test


def feature_engineer():
    train, label, test = load_data()

    # Fare
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    # Cabin
    train["Cabin"] = train["Cabin"].fillna("N")
    test["Cabin"] = test["Cabin"].fillna("N")

    # Embarked
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Embarked"] = test["Embarked"].fillna("S")

    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3

    train["A-embark"] = train["Embarked"].apply(getEmbark)
    test["A-embark"] = test["Embarked"].apply(getEmbark)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0

    train["A-name"] = train["Name"].apply(getName)
    test["A-name"] = test["Name"].apply(getName)

    feature = ["Survived", "Pclass", "SibSp", "Parch", "Fare", "A-embark", "A-name"]

    return train[feature], label, test[feature]


def stand_linear_regression(x, y):
    xMat = np.mat(x)
    yMat = np.mat(y).T

    xTx = xMat.T * xMat

    if np.linalg.det(xTx) == 0.0:
        print("This matrix ins singular, cannot to inverse")
        return
    ws = xTx.I * xMat.T * yMat
    return ws


def predict_age(x_train, y_train, x_valid, y_valid):
    ws = stand_linear_regression(x_train, y_train)
    x_valid_Mat = np.mat(x_valid)
    print(ws)
    y_pre = x_valid_Mat * ws
    print y_pre
    # 12.4654083464
    # A-e -> 11.6206955336
    # A-n -> 10.0781662156
    print(mean_absolute_error(y_valid, y_pre))


if __name__ == "__main__":
    train, label, test = feature_engineer()

    x_train, x_valid, y_train, y_valid = train_test_split(train, label, test_size=0.3, random_state=2017)

    predict_age(x_train, y_train, x_valid, y_valid)