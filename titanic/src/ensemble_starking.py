import sys
import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from ensemble import Ensemble
from feature_engineer import feature_engineer
from utils import submission


#
# load data
#
x_train, y_train, x_test = feature_engineer()

#
# Set model
#
svc = SVC()
rf = RandomForestClassifier()

from ensemble_util import Ensemble


def submit(pre):
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pre
    })
    print("write submit file: *.csv")
    submit_file = '../sub/{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
    submission.to_csv(submit_file, encoding="utf-8", index=False)

if __name__ == '__main__':
    train = pd.read_csv('../data/feature_train.csv')
    test = pd.read_csv('../data/feature_test.csv')

base_models = [svc, rf, xgb, lgb, rf, gbm]

stack = Ensemble(n_splits=8,
                 stacker=DecisionTreeClassifier(),
                 base_models=base_models)
# Stacker score: 0.8239 LB: 0.779

    clfs = [DecisionTreeClassifier(),
            XGBClassifier(n_estimators=100, max_depth=4, min_child_weight=2),
            RandomForestClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4, n_jobs=4),
            GradientBoostingClassifier(n_estimators=140, max_depth=4, min_samples_split=6, min_samples_leaf=4)]

    ensemble = Ensemble(clfs, rf)
    prediction = ensemble.fit_predict(x_train, y_train, x_test)
    submit(prediction)




