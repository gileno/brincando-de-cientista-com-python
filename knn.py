# coding=utf-8

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

def sample1():
    df = pd.read_csv("wine.data")

    test_idx = np.random.uniform(0, 1, len(df)) <= 0.3
    train = df[test_idx==True]
    test = df[test_idx==False]

    features = [
        'alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
        'Color intensity', 'Hue', 'OD280/OD315', 'Proline'
    ]

    results = []

    for i in range(1, 50, 3):
        clf = KNeighborsClassifier(n_neighbors=i)
        clf.fit(train[features], train['class'])
        preds = clf.predict(test[features])
        accuracy = np.where(preds==test['class'], 1, 0).sum() / float(len(test))
        print "Neighbors: %d, Accuracy: %3f" % (i, accuracy)
        results.append([i, accuracy])

    results = pd.DataFrame(results, columns=["n", "accuracy"])

    plt.plot(results.n, results.accuracy)
    plt.title("Accuracy with Increasing K")
    plt.show()

def sample2():
    df = pd.read_csv("wine.data")

    test_idx = np.random.uniform(0, 1, len(df)) <= 0.3
    train = df[test_idx==True]
    test = df[test_idx==False]

    features = [
        'alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
        'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
        'Color intensity', 'Hue', 'OD280/OD315', 'Proline'
    ]
    list_features = []
    list_features.append(generate_features_samples(features, 8))
    list_features.append(generate_features_samples(features, 8))
    list_features.append(generate_features_samples(features, 8))

    fig, ax = plt.subplots()

    for i, sample_features in enumerate(list_features):
        results = []
        print 'Sample %s - %s' % (i, sample_features)
        for n_neighbors in range(1, 50, 3):
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(train[sample_features], train['class'])
            preds = clf.predict(test[sample_features])
            accuracy = np.where(preds==test['class'], 1, 0).sum() / float(len(test))
            print "Sample Features: %d, Accuracy: %3f" % (i, accuracy)
            results.append([n_neighbors, accuracy])

        results = pd.DataFrame(results, columns=["n", "accuracy"])
        ax.plot(results.n, results.accuracy, label='Sample %s' % i)
    legend = ax.legend(loc='upper center', shadow=True)

    frame  = legend.get_frame()
    frame.set_facecolor('0.90')

    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.title("Accuracy with changing the features")
    plt.show()

def generate_features_samples(features, length=None):
    size = len(features)
    if length is None:
        length = (size / 2) + 1
    features_copy = list(features)
    random.shuffle(features_copy)
    return features_copy[:length]
