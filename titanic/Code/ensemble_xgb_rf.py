# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import cross_val_score


def stacking(alg, x_train, y_train, test):
    alg.fit(x_train, y_train)
    oof_y_train = alg.predict(x_train)
    oof_y_subminssion = alg.predict(test)

    return oof_y_train, oof_y_subminssion


def submit(alg, test_data):
    predict_data = alg.predict(test_data[features])
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predict_data
    })
    submission.to_csv('../sub/bagging.csv', index=False)


if __name__ == '__main__':
    train = pd.read_csv('../data/feature_train.csv')
    test = pd.read_csv('../data/feature_test.csv')
    print train.head()

    features = ["Pclass", "Fare", "Age", "SibSp", "child", "Parch", "sex", "fimalysize",
                "embark", "name", 'cabin']
    x_train = train[features]
    y_train = train["Survived"]
    print x_train.info()

    rf = RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4, n_jobs=4)
    rf_scores = cross_val_score(rf, x_train, y_train, cv=3)

    dt = DecisionTreeClassifier()
    dt_scores = cross_val_score(dt, x_train, y_train)
    dt.fit(x_train, y_train)

    xgb = XGBClassifier(n_estimators=140, max_depth=4, min_child_weight=6)
    xgb_scores = cross_val_score(xgb, x_train, y_train)
    xgb.fit(x_train, y_train)

    bagging_clf = BaggingClassifier(xgb, max_samples=0.9, max_features=1.0, bootstrap=True,
                                    bootstrap_features=False, n_jobs=4)
    bagging_scores = cross_val_score(bagging_clf, x_train, y_train, cv=3)
    bagging_clf.fit(x_train, y_train)

    print "rf scores:", rf_scores.mean()
    print "dt scores:", dt_scores.mean()
    print "xgb scores:", xgb_scores.mean()
    print "bagging scores:", bagging_scores.mean()

    submit(xgb, test)