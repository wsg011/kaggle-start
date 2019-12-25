import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load train and validation data set.
    parameter:
        no

    return:
        x_train y_train
        x_valid y_valid
        test
    """
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    train_df = train.drop(['label'], axis=1)
    label = pd.get_dummies(train['label'])
    train_df = train_df.applymap(lambda x: x / 255)
    test = test.applymap(lambda x: x / 255)

    x_train, x_valid, y_train, y_valid = train_test_split(train_df, label, test_size=0.2, random_state=2017)
    print("x_train shape:" + str(x_train.shape))
    print("x_valid shape:" + str(x_valid.shape))

    return x_train, y_train, x_valid, y_valid, test

x_train, y_train, x_valid, y_valid, x_test = load_data()

# Define tensorflow softmax model
sess =tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
yhat = tf.nn.softmax(tf.matmul(x, w) + b)

# cost function
cross_entropy = -tf.reduce_sum(y*tf.log(yhat))
# gradient descent optimizer, learn rate 0.01
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Evaluation model with validation data
correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# init variables
sess.run(tf.global_variables_initializer())

train_size = x_train.shape[0]
batch_size = 64

for i in range(1000):
    start = i * batch_size % train_size
    end = (i + 1) * batch_size % train_size
    # print(start, end)
    if start > end:
        start = 0

    batch_x = x_train[start:end]
    batch_y = y_train[start:end]

    if i % 100 == 0:
        print("train-loss ======== > {}".format(cross_entropy.eval(feed_dict={x: batch_x, y: batch_y})))
        print("train-accuracy ==== > {}".format(accuracy.eval({x: x_train, y: y_train})))
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

print("validation accuracy score is {}".format(accuracy.eval({x: x_valid, y: y_valid})))

# Prediction and submission
y_pre = sess.run(tf.nn.softmax(tf.matmul(x, w) + b), feed_dict={x:x_test})
pre_label = np.argmax(y_pre, axis=1)


submit_df = pd.read_csv('../input/sample_submission.csv')
submit_df.Label = pre_label
submit_df.to_csv('tensorflow_lr.csv', index=None, encoding='utf-8')
