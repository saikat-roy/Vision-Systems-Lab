# Vision Systems Lab: Learning Computer Vision on GPUs
Repository of the CudaVision Lab at University of Bonn (SS19) implemented (mostly) on PyTorch, Python3 and Jupyter notebooks. The project begins from the basics of neural networks and continues to deeper models. The following projects are contained in the respective folders:

### Project 1: Softmax Regression (without autograd/Pytorch Tensors)
Involves using softmax regression with manual gradient calculation for classifying the MNIST dataset. Training and test set accuracies after a simple 5 iteration run was `0.8931` and `0.8866` respectively.

### Project 2: Multilayer Neural Network
Involves training simple multilayer neural networks using vanilla SGD on PyTorch with k-fold monte-carlo cross validation for hyperparameter (learning rate and batch size) search. Classification was done on the CIFAR-10 dataset. A confusion matrix after a simple 50 iteration run on a `3072-128-128-10` architecture is given below, with a training and test set accuracy of `0.6647` and `0.5117` respectively.

![](https://github.com/saikat-roy/Vision-Systems-Lab/blob/master/Project2/conf_mat.png "Confusion Matrix after 50 iterations on a simple network")

### Project 3: Different Optimizers for MLP
The project involved applying `SGD`, `Adam`, `RMSprop`, `Adagrad` and `Adadelta` to the CIFAR-10 dataset. A `3072-128-128-10` architecture with 0.2 Dropout between hidden units was used for all algorithms. 

![](https://github.com/saikat-roy/Vision-Systems-Lab/blob/master/Project3/optims.png "Training with different non-linearities")

Also tested were different non-linearities for the hidden units.

![](https://github.com/saikat-roy/Vision-Systems-Lab/blob/master/Project3/nonlins.png "Training with different non-linearities")

In each case, the results were kind of counterintuitive as `SGD` and `sigmoid` performed the best. However, it might be possible that the convergence rates simply might be different. Additionally, with the network being this shallow, the benefits of the non-linearities used typically in 'Deep' networks might simply not reflect on a network of this scale. And if run long enough, `SGD` generally converges to a better minima than other optimizers like `Adam`.

### Project 4: Convolutional Neural Networks
The project introduces one to convolutional neural networks and applied a CNN on the CIFAR-10 dataset.
