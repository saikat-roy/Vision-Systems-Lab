import numpy as np
import numpy.matlib
import pickle

class SoftmaxRegression:

    def __init__(self, k, n_class):
        """
        Constructor for object initialization

        :param k: number of dimensions of the data
        :param n_class: number of classes/output variables for the problem
        """
        self.n_class = n_class
        self.W = np.random.rand(k, n_class)
        self.W = np.append(self.W, np.zeros((1, n_class)),0)

    def softmax(self, x, eps=0.0001):
        """
        Softmax Nonlinearity applied to x and returned. Does not do argmax
        :param x: 1 x k ndarray
        :return: softmax non-linearity applied to x
        """
        return np.exp(x) / (np.sum(np.exp(x))+eps)

    def fit(self, x, y, batch_size = 20, lr = 0.01, iters = 100):
        """
        Fit the softmax regression model to data. Final usage as classification or regression model irrelevant here.

        :param batch_size: batch size
        :param lr: learning rate
        :param x: n x k ndarray of data points for model training/fitting
        :param y: n x n_class ndarray of labels for fitting
        """
        print(x.shape)
        x = np.append(x, np.ones((x.shape[0],1)), 1)
        assert x.shape[0] == y.shape[0] # x and y should have same number of samples

        # Number of iterations
        for iter in range(iters):
            for i in range(0, max(1,x.shape[0]-batch_size)): # Iterate over training data

                # Make batches
                x_batch = x[i:min(i+batch_size,x.shape[0])]
                y_batch = y[i:min(i+batch_size,x.shape[0])]

                # Get prediction for batch
                y_hat_batch = self.predict(x_batch, probs=True, aug1=False)

                W_update = np.zeros_like(self.W)

                # Iterate over batch
                for j in range(batch_size):
                    # print(y_hat_batch.shape, y_batch.shape)
                    der_t1 = (y_batch[j] - y_hat_batch[j]) * y_hat_batch[j] * (1 - y_hat_batch[j])
                    # print(der_t1)
                    #exit()
                    f = lambda x: x*der_t1
                    # print(der_t1.shape, x_batch[j].shape)

                    x_rep = np.matlib.repmat(x_batch[j], self.n_class,1)
                    # print(x_rep.shape)
                    der_t2 = np.apply_along_axis(f, 0, x_rep)
                    # print(der_t2.shape)

                    # print(W_update.shape)
                    W_update = W_update + ((1/batch_size) * der_t2).T


                # der_t1 = (y_batch - y_hat_batch) * y_hat_batch * (1 - y_hat_batch)
                # f = lambda x: x * der_t1
                # x_rep = np.tile(x_batch, (x_batch.shape[0], x_batch.shape[1], self.n_class)) # n x k -> n x k x l
                # print(der_t1.shape, x_rep.shape)

                self.W = self.W + lr * W_update
                # print(self.W)
                preds = model.predict(train_x, False)
                print(calculate_acc(train_y, preds))

    def predict(self, x, probs = False, aug1 = True):
        """
        Predicts the softmax regression output as classification or regression outputs

        :param X: n x k ndarray for prediction
        :param probs: True to return probs
        :param aug1: True to augment a column of 1s for bias
        :return: Softmax activations or class predictions, depending on probs parameter.
        """
        if aug1:
            x = np.append(x, np.ones((x.shape[0], 1)), 1)
        y = np.dot(x,self.W)#+self.b
        # print(y)
        if probs:
            return np.apply_along_axis(self.softmax, 1, y)
        else:
            return np.argmax(np.apply_along_axis(self.softmax, 1, y), axis=1)


def calculate_acc(y, y_hat):
    """
    Calculates accuracy of prediction
    :param y: true labels
    :param y_hat: predicted labels
    :return: accuracy in the range [0,1]
    """
    return np.sum(1 * (y==y_hat))/y_hat.shape[0]


if __name__ == "__main__":

    import torchvision.transforms as transforms
    import torchvision.datasets as dsets

    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

    train_x = train_dataset.train_data.numpy()
    train_y = train_dataset.train_labels.numpy()

    test_x = test_dataset.test_data.numpy()
    test_y = test_dataset.test_labels.numpy()

    train_x = np.reshape(train_x, (60000,-1))
    train_y = np.reshape(test_x, (60000,-1))

    train_x = (train_x-np.mean(train_x,0))/(np.std(train_x,0)+0.00001)
    train_one_hot_targets = np.eye(max(train_y)+1)[np.reshape(train_y,-1)]

    test_x = (test_x - np.mean(test_x, 0)) / (np.std(test_x, 0) + 0.00001)
    test_one_hot_targets = np.eye(max(test_y) + 1)[np.reshape(test_y, -1)]

    #print(train_x.shape, train_y.shape)

    model = SoftmaxRegression(train_x.shape[1], train_one_hot_targets.shape[1])

    model.fit(train_x, train_one_hot_targets, iters=100, batch_size=20)


    print("Final Train Acc.")
    preds = model.predict(train_x, False)
    print(calculate_acc(train_y, preds))

    print("Final Test Acc")
    preds = model.predict(test_x, False)
    print(calculate_acc(test_y, preds))