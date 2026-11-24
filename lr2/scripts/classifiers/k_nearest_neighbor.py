import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute distances using two loops.
        """
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
        """
        Compute distances using one loop.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i, :] = np.sqrt(
                np.sum((self.X_train - X[i]) ** 2, axis=1)
            )

        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute distances without explicit loops.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y
        X_squared = np.sum(X ** 2, axis=1, keepdims=True)
        train_squared = np.sum(self.X_train ** 2, axis=1)

        dists_squared = (
            X_squared
            + train_squared
            - 2 * X.dot(self.X_train.T)
        )

        # Из-за погрешности вычислений иногда могут появиться
        # очень маленькие отрицательные значения.
        dists_squared = np.maximum(dists_squared, 0)

        dists = np.sqrt(dists_squared)

        return dists

    def predict_labels(self, dists, k=1):
        """
        Predict labels based on nearest neighbors.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)

        for i in range(num_test):

            # Индексы k ближайших соседей
            nearest_indices = np.argsort(dists[i])[:k]

            # Их классы
            closest_y = self.y_train[nearest_indices]

            # Подсчитываем количество каждого класса
            counts = np.bincount(closest_y.astype(int))

            # При одинаковом количестве np.argmax
            # выбирает меньший индекс класса.
            y_pred[i] = np.argmax(counts)

        return y_pred