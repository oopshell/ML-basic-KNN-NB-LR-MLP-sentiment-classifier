import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from collections import Counter
from random import random


def load_data_glove(path):
    df = pd.read_csv(path)
    X_df = df['tweet']
    y_df = df['sentiment']

    X_df = X_df.str.strip(' []')
    X_df = X_df.str.split(pat=',', expand=True)
    X = np.array(X_df, dtype=np.float)

    y = pd.Categorical(y_df, ordered=True).codes  # labels = {'neg': 0, 'neu': 1, 'pos': 2}
    y = np.array(y)

    return X, y


def weight_random(y_train, y_dev):
    # calculate the prior probability of classes in the training set
    label_counter_train = Counter(y_train)
    prior_neg = label_counter_train[0] / len(y_train)
    prior_neu = label_counter_train[1] / len(y_train)
    prior_pos = label_counter_train[2] / len(y_train)

    # Construct baseline
    baseline = []
    for i in range(len(y_dev)):
        r = random()
        if r < 1 * prior_neg:
            class_i = 0  # 'neg'
        elif 1 * prior_neg <= r < 1 * prior_neg + 1 * prior_neu:
            class_i = 1  # 'neu'
        else:
            class_i = 2  # 'pos'
        baseline.append(class_i)

    return baseline


def evaluation(y_true, y_pred):  # , y_pred_prob
    acc = accuracy_score(y_true, y_pred)
    error = 1 - acc
    f1_ma = f1_score(y_true, y_pred, average='macro')
    f1_mi = f1_score(y_true, y_pred, average='micro')
    conf_mtx = confusion_matrix(y_true, y_pred)
    print("accuracy_score:", acc)
    print("error rate:", error)
    print("f1_macro:", f1_ma)
    print("f1_micro:", f1_mi)
    print("confusion_matrix:", conf_mtx)


if __name__ == "__main__":
    xx_train, yy_train = load_data_glove('data/train_glove.csv')
    xx_dev, yy_dev = load_data_glove('data/dev_glove.csv')
    wr_bl = weight_random(yy_train, yy_dev)
    evaluation(yy_dev, wr_bl)
