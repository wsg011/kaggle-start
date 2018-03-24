import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC

from feature_engineer import feature_engineer
from utils import submission


#
# Load data
#
train, label, test = feature_engineer()

x_train, x_valid, y_train, y_valid = train_test_split(train, label, test_size=0.2, random_state=0)
print("train shape:{}, test shape:{}".format(x_train.shape, x_valid.shape))

svc = SVC()

#
# cv
#
#start_time = time.time()
#print(cross_val_score(svc, train, label, cv=5, n_jobs=-1))
# score:[ 0.94122546  0.93538022  0.93808787  0.93902584  0.94104335]
#print("CV time:{}".format(time.time()-start_time))

start_time = time.time()
svc.fit(train, label)
print("fit time:{}".format(time.time()-start_time))
pre = svc.predict(test)
submission(pre)



