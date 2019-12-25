# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


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