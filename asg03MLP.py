# import tensorflow as tf
from tensorflow import keras
# from keras.models import Sequential
# from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.feature_selection import SelectPercentile
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
import numpy as np
import pandas as pd
import re
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
    train_df = pd.read_csv(path)
    X_df = train_df['tweet']
    y_df = train_df['sentiment']

    X_df = X_df.str.strip(' []')
    X_df = X_df.str.split(pat=',', expand=True)
    X = np.array(X_df, dtype=np.float)

    y = pd.Categorical(y_df, ordered=True).codes  # labels = {'neg': 0, 'neu': 1, 'pos': 2}
    y = np.array(y)

    return X, y


def model_mlp(x_train, y_train, x_dev, y_dev, x_test):
    # Encode class label
    y_train = to_categorical(y_train, 3)
    y_dev = to_categorical(y_dev, 3)
    # Define the NN model
    mlp = keras.Sequential(
        [
            # keras.layers.Input((5000, )),  # model for tfidf and count datasets
            # keras.layers.Dense(500, activation='relu'),
            # keras.layers.Dense(200, activation='relu'),
            # keras.layers.Dense(100, activation='relu'),
            # keras.layers.Dense(50, activation='relu'),
            # keras.layers.Dense(10, activation='relu'),
            # keras.layers.Dense(3, activation='softmax')

            keras.layers.Input((100,)),  # model for glove dataset
            keras.layers.Dense(100, activation='relu'),
            keras.layers.Dense(80, activation='relu'),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dense(20, activation='relu'),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(3, activation='softmax')
        ]
    )

    # Compile the model
    time_start = time()
    mlp.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    mlp.fit(x_train, y_train, epochs=10, batch_size=100)
    print("Training time:", round(time() - time_start, 4), "s")

    print(mlp.summary())
    # plot_model(mlp, to_file='pred_results/mlp_plot.png', show_shapes=True, show_layer_names=True)

    time_start = time()
    # Predict on dev set
    mlp_predictions_prob = mlp.predict(x_dev)
    print("Prediction time:", round(time() - time_start, 4), "s")
    mlp_prediction = np.argmax(mlp_predictions_prob, axis=1)
    loss, mlp_accuracy = mlp.evaluate(x_dev, y_dev)
    print("mlp_Accuracy:", mlp_accuracy)

    # Save predictions on test set
    test_predictions_prob = mlp.predict(x_test)
    test_predictions = np.argmax(test_predictions_prob, axis=1)
    test_predictions_df = pd.DataFrame(test_predictions)
    test_predictions_df.to_csv('pred_results/mlp_predictions.csv', index=False)

    return mlp_prediction


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
    xx_train, yy_train = load_data_glove('data/train_glove.csv')
    xx_dev, yy_dev = load_data_glove('data/dev_glove.csv')
    xx_test, yy_test = load_data_glove('data/test_glove.csv')
    mlp_pred = model_mlp(xx_train, yy_train, xx_dev, yy_dev, xx_test)

    # Train on TFIDF dataset
    # xx_train, yy_train = load_data_tfidf('data/train_tfidf.csv')
    # xx_dev, yy_dev = load_data_tfidf('data/dev_tfidf.csv')
    # xx_test, yy_test = load_data_tfidf('data/test_tfidf.csv')
    # mlp_pred = model_mlp(xx_train, yy_train, xx_dev, yy_dev, xx_test)
    # # Feature selection based on percentile
    # # fea_sel = SelectPercentile()  # score_func=f_classif, percentile=10
    # # fea_sel.fit(xx_train, yy_train)
    # # xx_train_sel = fea_sel.transform(xx_train)
    # # xx_dev_sel = fea_sel.transform(xx_dev)
    # # xx_test_sel = fea_sel.transform(xx_test)
    # # mlp_pred = model_mlp(xx_train_sel, yy_train, xx_dev_sel, yy_dev, xx_test_sel)

    # Train on Word Count dataset
    # xx_train, yy_train = load_data_tfidf('data/train_count.csv')
    # xx_dev, yy_dev = load_data_tfidf('data/dev_count.csv')
    # xx_test, yy_test = load_data_tfidf('data/test_count.csv')
    # mlp_pred = model_mlp(xx_train, yy_train, xx_dev, yy_dev, xx_test)

    # Evaluation Metrics
    evaluation(yy_dev, mlp_pred)
