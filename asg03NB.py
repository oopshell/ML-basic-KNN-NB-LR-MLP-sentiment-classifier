import numpy as np
import pandas as pd
import re
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_selection import SelectPercentile
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from time import time


def load_data_tfidf(path):
    df = pd.read_csv(path)
    X_df = df['tweet']
    y_df = df['sentiment']

    X_df = X_df.str.strip(' []')
    X_ = []
    for s in X_df:
        group = re.split("\(|\),\s\(|\)", s)
        group = list(filter(None, group))
        X_row = {}
        for tuple_str in group:
            t = tuple_str.split(', ')
            id = int(t[0])
            tfidf = float(t[1])
            X_row[id] = tfidf
        X_.append(X_row)

    X = np.zeros((len(X_), 5000), dtype=float)
    for i in range(len(X_)):
        t = X_[i]
        for wid in t:
            X[i, wid] = t[wid]

    y = pd.Categorical(y_df, ordered=True).codes  # labels = {'neg': 0, 'neu': 1, 'pos': 2}
    y = np.array(y)

    return X, y


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


def model_nb(x_train, y_train, x_dev, y_dev, x_test):
    time_start = time()
    # Build NB model
    # nb = GaussianNB()
    nb = MultinomialNB()
    # Train model
    nb.fit(x_train, y_train)
    print("Training time:", round(time() - time_start, 4), "s")
    time_start = time()
    # Predict on dev set
    nb_predictions = nb.predict(x_dev)
    print("Prediction time:", round(time() - time_start, 4), "s")
    nb_accuracy = nb.score(x_dev, y_dev)
    print('nb_Accuracy:', nb_accuracy)

    # Save predictions on test set
    test_predictions = nb.predict(x_test)
    test_predictions_df = pd.DataFrame(test_predictions)
    test_predictions_df.to_csv('pred_results/nb_predictions.csv', index=False)

    return nb_predictions


def evaluation(y_true, y_pred):
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
    # Train on GloVe dataset
    # xx_train, yy_train = load_data_glove('data/train_glove.csv')
    # xx_dev, yy_dev = load_data_glove('data/dev_glove.csv')
    # xx_test, yy_test = load_data_glove('data/test_glove.csv')
    # nb_pred = model_nb(xx_train, yy_train, xx_dev, yy_dev, xx_test)

    # Train on TFIDF dataset
    xx_train, yy_train = load_data_tfidf('data/train_tfidf.csv')
    xx_dev, yy_dev = load_data_tfidf('data/dev_tfidf.csv')
    xx_test, yy_test = load_data_tfidf('data/test_tfidf.csv')
    nb_pred = model_nb(xx_train, yy_train, xx_dev, yy_dev, xx_test)
    # Feature selection based on percentile
    # fea_sel = SelectPercentile()  # score_func=f_classif, percentile=10
    # fea_sel.fit(xx_train, yy_train)
    # xx_train_sel = fea_sel.transform(xx_train)  # .toarray()
    # xx_dev_sel = fea_sel.transform(xx_dev)  # .toarray()
    # xx_test_sel = fea_sel.transform(xx_test)
    # nb_pred = model_nb(xx_train_sel, yy_train, xx_dev_sel, yy_dev, xx_test_sel)

    # Train on Word Count dataset
    # xx_train, yy_train = load_data_tfidf('data/train_count.csv')
    # xx_dev, yy_dev = load_data_tfidf('data/dev_count.csv')
    # xx_test, yy_test = load_data_tfidf('data/test_count.csv')
    # nb_pred, nb_pred_prob = model_nb(xx_train, yy_train, xx_dev, yy_dev, xx_test)

    # Evaluation Metrics
    evaluation(yy_dev, nb_pred)
