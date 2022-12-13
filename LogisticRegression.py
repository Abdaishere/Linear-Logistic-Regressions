import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression


def fit(features, output, iterations, alpha):
    row, col = features.shape
    theta_1 = np.zeros(col)
    theta_0 = 0

    for i in range(iterations):
        theta_0, theta_1 = gradientDescent(theta_1, theta_0, features, output, alpha, row)
    return theta_0, theta_1


def gradientDescent(theta_1, theta_0, features_data, output_data, alpha, rows):
    equation = 1 / (1 + np.exp(- (features_data.dot(theta_1) + theta_0)))

    tmp = (equation - output_data.T)
    tmp = np.reshape(tmp, rows)
    dTheta_1 = np.dot(features_data.T, tmp) / rows
    dTheta_0 = np.sum(tmp) / rows

    theta_1 -= alpha * dTheta_1
    theta_0 -= alpha * dTheta_0

    return theta_0, theta_1


def predict(x, th1, th0):
    eq = 1 / (1 + np.exp(- (x.dot(th1) + th0)))
    out = np.where(eq > 0.5, 1, 0)
    return out


def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


def main():
    data_frame = pd.read_csv("customer_data.csv")
    data_frame = shuffle(data_frame)
    x = data_frame.iloc[:, :-1].values
    y = data_frame.iloc[:, -1:].values
    # Normalized
    normalized_data = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    X_train, X_test, Y_train, Y_test = train_test_split(
        normalized_data, y, test_size=1 / 3, random_state=0)

    # training

    theta_0, theta_1 = fit(X_train, Y_train, 1000, 0.1)
    model1 = LogisticRegression()
    model1.fit(X_train, Y_train)

    Y_pred = predict(X_test, theta_1, theta_0)
    Y_pred1 = model1.predict(X_test)

    print("Accuracy on test set	: ", accuracy(y_pred=Y_pred, y_test=Y_test))
    print("Accuracy on test set by sklearn model : ", accuracy(y_pred=Y_pred1, y_test=Y_test))



main()
