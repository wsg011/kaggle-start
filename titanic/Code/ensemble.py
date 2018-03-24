# -*- encoding:utf-8 -*-

import numpy as np
import pandas as pd
# import models
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


class Ensemble(object):
    def __init__(self, base_models, sec_model, n_folds=5):
        self.n_folds = n_folds
        self.base_models = base_models
        self.clf = sec_model

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)

        folds = KFold(n_splits=self.n_folds, random_state=0)
        en_train = np.zeros((X.shape[0], len(self.base_models)))
        en_test = np.zeros((T.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            en_test_i = np.zeros((T.shape[0], self.n_folds))
            print("fit model:%s " % str(i+1))
            for j, (train_idx, test_idx) in enumerate(folds.split(X, y)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pre = clf.predict(X_holdout)
                en_train[test_idx, i] = y_pre
                en_test_i[:, j] = clf.predict(T)[:]
            score = accuracy_score(en_train[:, i], y)
            print("model %s scoring: %s" % (i, score))
            en_test[:, i] = en_test_i.mean(1)
        self.clf.fit(en_train, y)
        pre = self.clf.predict(en_test)
        return pre

    def predict(self, x):
        return self.clf.predict(x)

    def score(self, x, y):
        s = accuracy_score(y, self.predict(x))
        return s

if __name__ == "__main__":
    print("load data...")
    train = pd.read_csv("../data/feature_train.csv")
    test = pd.read_csv("../data/feature_test.csv")
    features = ["Pclass", "Age", "sex", "child", "fimalysize", "Fare", "embark", "cabin", "name"]
    x_train, y_train = train[features], train['Survived']
    x_test = test[features]

    lr = LogisticRegression()
    svc = SVC()
    dt = DecisionTreeClassifier()
    et = ExtraTreesClassifier()
    ada = AdaBoostClassifier()
    rf = RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_leaf=4, min_samples_split=6)
    GBDT = GradientBoostingClassifier()
    xgb_GBDT = XGBClassifier(objective='binary:logistic', learning_rate=0.1, n_estimators=40, max_depth=6)

    clfs = [lr, et, ada, rf, GBDT, xgb_GBDT]
    ensemble = Ensemble(clfs)
    sec_train, sec_test = ensemble.fit_predict(x_train, y_train, x_test)
                y_pred = clf.predict(X_holdout)[:]
                print ("Fit Model %d fold %d: %s" % (i, j, accuracy_score(y_holdout, y_pred)))
    clf = lr
    clf.fit(sec_train, y_train)
    #
    #score = 0
    #for i in range(0, 10):
    #    num_test = 0.2
    #    X_train, X_cv, Y_train, Y_cv = train_test_split(x_train, y_train, test_size=num_test)
    #    ensemble.fit(X_train, Y_train)
    #    # Y_test = bag.predict(X_test)
    #    acc_xgb = round(ensemble.score(X_cv, Y_cv) * 100, 2)
    #    score += acc_xgb
    #print(score / 10)  # 0.8786
    pre = clf.predict(sec_test)

    predict_dataframe = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pre.astype(int)
    })
    predict_dataframe.to_csv('../data/ensemble.csv', index=False, encoding="utf-8")

