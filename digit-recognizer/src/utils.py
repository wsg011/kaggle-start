import numpy as np
import pandas as pd
from datetime import datetime


def load_data():
    print("load data...")
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    return train, test


def submission(pre):
    sample = pd.read_csv('../input/sample_submission.csv')

    sample['Label'] = pre

    submit_file = '../sub/{}.csv'.format(datetime.now().strftime('%Y%m%d_%H_%M'))
    sample.to_csv(submit_file, index=False)