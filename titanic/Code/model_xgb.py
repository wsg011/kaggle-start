# -*- encoding:utf-8 -*--

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV


print "load data..."
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print "train data set:", train.shape
print "test date set:", test.shape

print "clean data..."

def clean_data(titanic):
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

    titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
    titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

    titanic.loc[titanic["Embarked"] == 'S', "Embarked"] = 1
    titanic.loc[titanic["Embarked"] == 'C', "Embarked"] = 2
    titanic.loc[titanic["Embarked"] == 'Q', "Embarked"] = 3
    titanic.loc[titanic["Embarked"].isnull()] = 1

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic

train_data = clean_data(train)
test_data = clean_data(train)

# Engineer Features
print "Engineer Feature..."
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# Model of Random Forest
print "fit model..."
xgb = XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    nthread=4,
    seed=100,
)
params = {
    'max_depth': range(3, 11, 2)
}
grid_Search = GridSearchCV(estimator=xbg, param_grid=params, cv=5)
grid_Search.fit(train_data[predictors], train_data['survived'])
grid_Search.grid_scores_, grid_Search.best_params_, grid_Search.best_score_




