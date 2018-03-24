import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from utils import load_data


def feature_engineer():
    train, test = load_data()

    print("feature transform...")
    x_train = train.drop(['label'], axis=1)
    x_train = x_train.applymap(lambda x: x/255.0)

    label = train['label'].values

    #encode = OneHotEncoder()
    #y_train = encode.fit_transform(train['label'].reshape(-1, 1)).toarray()

    test = test.applymap(lambda x: x/255.0)

    return x_train, label, test