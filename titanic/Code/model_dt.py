# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import tree


def submit(pre):
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predict_data
    })
    print "write submit file:decisiontree.csv"
    submission.to_csv('../../sub/decisiontree.csv', index=False)


if __name__ == "__main__":
    print "load data..."
    train = pd.read_csv("../../data/feature_train.csv")
    test = pd.read_csv("../../data/feature_test.csv")
    print train.info(), test.info()

    # Select Features
    predictors = ["Pclass", "sex", "Age", "Fare", "embark"]
    print "feature select:", predictors

    print "Fix model..."
    # Model of DecisionTree
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(train[predictors], train["Survived"])

    print "Predict in test data set..."
    predict_data = dt.predict(test[predictors])
    submit(predict_data)

