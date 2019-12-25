# -*- encoding:utf-8 -*-
# !/usr/bin/python

import numpy as np
import pandas as pd
from datetime import datetime


#
# load data
#
def load_data():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    submission = pd.read_csv('../input/submission.csv')

    return train, test, submission

#
# submit file
#
def submission(pre):
    test = pd.read_csv('../input/test.csv')
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": pre
    })

    submit_file = '../sub/{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
    print("write submit file:{}".format(submit_file))
    submission.to_csv(submit_file, encoding="utf-8", index=False)