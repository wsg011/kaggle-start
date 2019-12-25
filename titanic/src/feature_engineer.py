# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils import load_data


#
# Feature Engineering
#
def feature_engineer():
    train, test = load_data()

    #
    # clean data
    # Age
    train["Age"] = train["Age"].fillna(train["Age"].median())
    test["Age"] = test["Age"].fillna(test["Age"].median())

    # Fare
    train["Fare"] = train["Fare"].fillna(train["Fare"].median())
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    # Cabin
    train["Cabin"] = train["Cabin"].fillna("N")
    test["Cabin"] = test["Cabin"].fillna("N")

    # Embarked
    train["Embarked"] = train["Embarked"].fillna("S")
    test["Embarked"] = test["Embarked"].fillna("S")

    #
    # feature engineer
    # sex
    train["A-sex"] = train["Sex"].apply(lambda x: 1 if x == "male" else 0)
    test["A-sex"] = test["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # child
    train["A-child"] = train["Age"].apply(lambda x: 1 if x < 16 else 0)
    test["A-child"] = test["Age"].apply(lambda x: 1 if x < 16 else 0)

    # older
    train["A-older"] = train["Age"].apply(lambda x: 1 if x > 45 else 0)
    test["A-older"] = test["Age"].apply(lambda x: 1 if x > 45 else 0)

    # familysize
    train["A-fimalysize"] = train["SibSp"] + train["Parch"] + 1
    test["A-fimalysize"] = test["SibSp"] + test["Parch"] + 1

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

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    train["A-cabin"] = train["Cabin"].apply(getCabin)
    test["A-cabin"] = test["Cabin"].apply(getCabin)
    features = ["Pclass", "Fare", "Age", "SibSp", "Parch", "A-child", "A-older", "A-sex", "A-fimalysize",
                "A-embark", "A-name", 'A-cabin']

    x_train = train[features]
    y_train = train['Survived'].values
    x_test = test[features]

    return x_train, y_train, x_test




