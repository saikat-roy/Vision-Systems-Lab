import numpy as np

class SoftmaxRegression:

    def __init__(self, k, n_class):
        """
        Constructor for object initialization

        :param k: number of dimensions of the data
        :param n_class: number of classes/output variables for the problem
        """
        self.W = np.random.rand(k, n_class)
        self.b = np.zeros((1, n_class))

    def softmax(self, x):
        """
        Softmax Nonlinearity applied to x and returned. Does not do argmax
        :param x: 1 x k ndarray
        :return: softmax non-linearity applied to x
        """
        return np.exp(x) / np.sum(np.exp(x))

    def fit(self, x, y, batch_size=20):
        """
        Fit the softmax regression model to data. Final usage as classification or regression model irrelevant here.

        :param x: n x k ndarray of data points for model training/fitting
        :param y: n x n_class ndarray of data points for fitting
        """

    def predict(self, x, probs = False):
        """
        Predicts the softmax regression output as classification or regression outputs

        :param X: n x k ndarray for prediction
        :param probs: True to return probs
        :return: Softmax activations or class predictions, depending on probs parameter.
        """
        y = np.dot(x,self.W)+self.b
        print(y)
        if probs:
            return np.apply_along_axis(self.softmax, 1, y)
        else:
            return np.argmax(np.apply_along_axis(self.softmax, 1, y), axis=1)

if __name__ == "__main__":

    model = SoftmaxRegression(3, 2)
    a = np.array([[1, 2, 3], [5, 6, 7]])

    print(model.predict(a, True))