# -*- coding: utf-8 -*-
import pandas as pd


def load_data(path="../input/"):
    train = pd.read_csv(path+'train.csv')
    test = pd.read_csv(path+'test.csv')
    submission = pd.read_csv(path+'gender_submission.csv')
    
    return train, test, submission