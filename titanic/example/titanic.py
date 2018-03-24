# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import tree

if __name__ == "__main__":
    print "load data..."
    train = pd.read_csv("../data/train.csv")
    test = pd.read_csv("../data/test.csv")
    print train.info(), test.info()

    print "clean data..."
    # Age
    train["Age"] = train["Age"].fillna(train["Age"].median())
    test["Age"] = test["Age"].fillna(test["Age"].median())
    # Fare
    train["Sex"] = train["Sex"].apply(lambda x: 1 if x == "male" else 0)
    test["Sex"] = test["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # Select Features
    feature = ["Age", "Sex"]

    print "Fix model..."
    # Model of DecisionTree
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(train[feature], train["Survived"])

    print "Predict in test data set..."
    predict_data = dt.predict(test[feature])

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_data
    })
    print "write submit file:decision_tree.csv"
    submission.to_csv('../data/decision_tree.csv', index=False)

