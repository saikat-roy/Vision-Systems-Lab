# Vision Systems Lab: Learning Computer Vision on GPUs
Repository of the CudaVision Lab at University of Bonn (SS19) implemented (mostly) on PyTorch, Python3 and Jupyter notebooks. The project begins from the basics of neural networks and continues to deeper models. The following projects are contained in the respective folders:

### Project 1: Softmax Regression (without autograd/Pytorch Tensors)
Involves using softmax regression with manual gradient calculation for classifying the MNIST dataset. Training and test set accuracies after a simple 5 iteration run was `0.8931` and `0.8866` respectively.

### Project 2: Multilayer Neural Network
Involves training simple multilayer neural networks on PyTorch with k-fold cross validation for hyperparameter search. Classification was done on the CIFAR-10 dataset. A confusion matrix after a simple 50 iteration run on a `3072-128-128-10` architecture is given below, with a training and test set accuracy of `0.6647` and `0.5117` respectively.

![](https://github.com/saikat-roy/Vision-Systems-Lab/blob/master/Project2/conf_mat.png "Confusion Matrix after 50 iterations on a simple network")
