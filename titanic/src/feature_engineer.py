# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from utils import load_data


def feature_engineer(df, features=None):
    """ 特种工程
    特征工程包括以下的工作：
        1.缺失值填充
        2.OneHot编码
    params:
        df: 输入数据
        features: 输出的特征列表
    return：
        feature_df: 输出特征
            columns:['age', 'fare', 'sex', 'child', 'older', 'fimalysize', 'embarked',
                     'embark', 'name', 'cabin']

    """
    feature_df = pd.DataFrame()

    # Age
    feature_df["age"] = df["Age"].fillna(df["Age"].median())

    # Fare
    feature_df["fare"] = df["Fare"].fillna(df["Fare"].median())

    # sex
    feature_df["sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # child
    feature_df["child"] = df["Age"].apply(lambda x: 1 if x < 16 else 0)

    # older
    feature_df["older"] = df["Age"].apply(lambda x: 1 if x > 45 else 0)

    # familysize
    feature_df["SibSp"] = df["SibSp"]
    feature_df["Parch"] = df["Parch"]
    feature_df["fimalysize"] = df["SibSp"] + df["Parch"] + 1

    # embark
    feature_df["embark"] = df["Embarked"].fillna("S")
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3
    feature_df["embark"] = feature_df["embark"].apply(getEmbark)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0
    feature_df["name"] = df["Name"].apply(getName)

    # cabin
    feature_df["cabin"] = df["Cabin"].fillna("N")
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1
    feature_df["cabin"] = feature_df["cabin"].apply(getCabin)

    return feature_df




