# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier

from load_data import load_data
from feature_engineer import feature_engineer


if __name__ == "__main__":
    train_df, test_df, submission = load_data()

    x_train = feature_engineer(train_df)
    y_train = train_df["Survived"]
    x_test = feature_engineer(test_df)

    rf = RandomForestClassifier(
        n_estimators=500,
        min_samples_split=12,
        min_samples_leaf=1,
        oob_score=True,
        random_state=2019
    )

    # train
    rf.fit(x_train,y_train)
    y_pre = rf.predict(x_test)

    submission["Survived"] = y_pre
    submission.to_csv("../output/submission.csv", index=False, encoding="utf-8")