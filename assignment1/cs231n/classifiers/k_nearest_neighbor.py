from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train, self.y_train = X, y

    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                dists[i, j] = np.sqrt(
                    np.sum((X[i] - self.X_train[j]) ** 2)
                )

        return dists

    def compute_distances_one_loop(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i] = np.sqrt(
                np.sum((self.X_train - X[i]) ** 2, axis=1)
            )

        return dists

    def compute_distances_no_loops(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        dists = np.sqrt(
            np.maximum(
                np.sum(X ** 2, axis=1, keepdims=True)
                + np.sum(self.X_train ** 2, axis=1)
                - 2 * X.dot(self.X_train.T),
                0
            )
        )

        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):

            # Find indices of k nearest training points
            closest_y = self.y_train[
                np.argsort(dists[i])[:k]
            ]

            # Find most common label.
            # np.bincount counts occurrences of each integer label.
            # np.argmax returns the first maximum, so ties go
            # to the smaller label.
            y_pred[i] = np.argmax(
                np.bincount(closest_y.astype(int))
            )

        return y_pred