# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

train = pd.read_csv("data/train.csv", dtype={"Age": np.float64},)
test = pd.read_csv("data/test.csv", dtype={"Age": np.float64},)


def harmonize_data(titanic):
    #填充空数据 和 把string数据转成integer表示
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
    titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = harmonize_data(train)
test_data = harmonize_data(test)

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

lr = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(
    lr,
    train_data[predictors],
    train_data["Survived"],
    cv=10
)

print scores.mean()
