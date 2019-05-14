import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import dataloader, random_split

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid

from Project2.sklearn_confusion_matrix import plot_confusion_matrix

class MLP(nn.Module):

    def __init__(self, n_input, n_hidden_layers, n_output, h_units):
        """
        Initialization for a simply softmax regression MLP model with ReLU activations in hidden layers
        :param n_input (int): Number of input units to network
        :param n_hidden_layers (int): Number of hidden layers in network
        :param n_output (int): Number of output units of network
        :param h_units (int or list): hidden unit count or list of hidden units in each hidden layer of network
        """
        super(MLP, self).__init__()
        self.n_out = n_output

        layers = []

        def add_layer_and_act(n_inp, n_out, nl_type):
            if nl_type is None:
                return [nn.Linear(n_inp, n_out)]
            return [nn.Linear(n_inp, n_out), self.non_lin(nl_type)]

        # Add input layers
        if type(h_units) is list:
            layers.extend(add_layer_and_act(n_input, h_units[0], 'relu'))
        else:
            layers.extend(add_layer_and_act(n_input, h_units, 'relu'))

        # Add hidden layers
        if n_hidden_layers>1:
            for i in range(1, n_hidden_layers):
                if type(h_units) is list:
                    layers.extend(add_layer_and_act(h_units[i-1], h_units[i], 'relu'))
                else:
                    layers.extend(add_layer_and_act(h_units, h_units, 'relu'))

        # Add output layer
        if type(h_units) is list:
            layers.extend(add_layer_and_act(h_units[-1], self.n_out, None))
        else:
            layers.extend(add_layer_and_act(h_units, self.n_out, None))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        """
        Simple forward pass
        :param x:
        :return:
        """
        return self.block(x)

    def non_lin(self, nl_type='sigmoid'):
        """
        Simply plugs in a predefined non-linearity from a dictionary to be used throughout the network
        :param nl_type: type based on predefined types. Defaults to sigmoid on wrong type.
        :return:
        """
        nl = {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(), 'softmax': nn.Softmax(self.n_out)}
        try:
            return nl[nl_type]
        except:
            print("non linearity type not found. Defaulting to sigmoid.")
            return


def train(dataloader, iters = 20):
    """
    Trains the model on the given dataloader and returns the loss per epoch
    :param dataloader: The autoencoder is trained on the dataloader
    :param iters: iterations for training
    :return:
    """

    loss_l = []
    for itr in range(iters):
        av_itr_loss = 0.0
        for batch_id, (x, y) in enumerate(dataloader):
            optimizer.zero_grad()
            # print(x)
            x = x.cuda()
            y = y.cuda()
            # x = (x>0.5).float() * 1
            x = x.view(batch_size, -1)
            # print(x[0])
            preds = model(x)
            # print((z==1).sum())
            batch_loss = loss(preds, y)
            batch_loss.backward()
            optimizer.step()
            av_itr_loss += (1/batch_size)*batch_loss.item()
        loss_l.append(av_itr_loss)
        print("Epoch {}: Loss={}".format(itr, av_itr_loss))
    return loss_l


def acc(dataloader):
    """
    Calculate accuracy of predictions from model for dataloader.
    :param dataloader: dataloader to evaluate
    :return:
    """
    acc = 0.0
    true_y = []
    pred_y = []
    total = 0.0
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            x = x.view(batch_size, -1)
            # print(x[0])
            preds = model(x)
            preds = torch.argmax(preds, dim=1)
            acc += ((preds==y).sum().item())
            total+= y.size(0)

            true_y.extend(list(preds.view(-1).cpu().numpy()))
            pred_y.extend(list(y.view(-1).cpu().numpy()))

        acc/=total
    return true_y, pred_y, acc


if __name__ == "__main__":

    batch_size = 64
    n_itr = 50

    model = MLP(1024 * 3, 1, 10, 100).cuda()
    print(model)

    transform_list = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_list)
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_list)

    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader  = torch.utils.data.DataLoader(testset,  batch_size=batch_size, shuffle=False, drop_last=True)

    print("\nTraining on Cross Entropy Loss:")
    # Using Cross Entropy Error as in Question 4
    lr = 0.001  # Need a higher lr for CE to converge
    loss_type = "Cross_Entropy"
    model.train()
    # loss = lambda x, z: (x*torch.log1p(z) + (1-x) * torch.log1p(1-z)).sum(dim=1).mean()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_list = train(train_dataloader, iters=n_itr)

    plt.plot(np.linspace(1,n_itr,n_itr),loss_list)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss Vs Iterations")
    plt.savefig("loss.png", format='png', transparent=False, pad_inches=0.1)
    plt.show()

    preds, true, train_acc = acc(train_dataloader)
    print("Train Acc={}".format(train_acc))

    preds, true, test_acc = acc(test_dataloader)
    print("Test Acc={}".format(test_acc))

    plot_confusion_matrix(true, preds, [str(i) for i in range(10)], normalize=True,
                          title="Confusion Matrix: CIFAR-10 (Test)")

    # Reinitializing dataloader with best batch size
    test_dataloader  = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=True)
    preds, true, test_acc = acc(test_dataloader)
    print("Accuracy on Test Set={}".format(test_acc))
