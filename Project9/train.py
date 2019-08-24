import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import dataloader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import cv2
import time

import sys

sys.path.append("E:/Vision-Systems-Lab/Project9/")

from model import *
from utils import *
import random
random.seed(12345)

def train(train_dataloader, valid_dataloader, iters=20, suppress_output=False,
          model_save_path="best.pth"):
    """
    Trains the model on the given dataloader and returns the loss per epoch
    """
    loss_l = []
    train_acc_l = []
    valid_acc_l = []
    best_valid_acc = 0.0
    equiv_train_acc = 0.0
    for itr in range(iters):
        av_itr_loss = 0.0
        model.train()
        for batch_id, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            preds = model(x)

            y = F.pad(y, (4, 5, 7, 6, 0, 0, 0, 0), mode='constant', value=0)
            batch_loss = loss(preds, y) #????
            batch_loss.backward()
            optimizer.step()
            av_itr_loss += (1 / y.size(0)) * batch_loss.item()
        loss_l.append(av_itr_loss)
        _, _, train_acc = acc(train_dataloader)
        _, _, valid_acc = acc(valid_dataloader)
        if not suppress_output:
            if itr % 1 == 0 or itr == iters - 1:
                print("Epoch {}: Loss={}, Training Accuracy:{}, Validation Accuracy:{}"
                      .format(itr, av_itr_loss, train_acc, valid_acc))
        train_acc_l.append(train_acc)
        valid_acc_l.append(valid_acc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            equiv_train_acc = train_acc
            torch.save(model.state_dict(), model_save_path)

    model.load_state_dict(torch.load(model_save_path))

    #   return loss_l, train_acc_l, valid_acc_l
    return loss_l, equiv_train_acc, best_valid_acc

def get_boxes(img, threshold):

    canny_output = cv2.Canny(img, threshold, threshold * 2)

    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    return boundRect

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
    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()

            preds = model(x)
            #preds = torch.argmax(preds, dim=1)  # ??
            #For each channel
            acc = np.zeros(preds.shape[1])
            for chan in range(preds.shape[1]):
                # Erotion
                kernel = np.ones((5, 5), np.uint8)
                preds_erosion = cv2.erode(np.array(preds.cpu())[0,chan], kernel, iterations=1)

                # Dilation
                preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                """                                                                                 ???
                After thresholding each
                output channel, we apply morphological erosion and dilation
                to eliminate negligible responses. Finally, we compute the
                object center coordinates from the achieved contours.
                """

                # Contour Detection

                (thresh, preds_dilation) = cv2.threshold(preds_dilation, 10, 200, 0)
                contours_p, _ = cv2.findContours(preds_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                contours_t, _ = cv2.findContours(np.array(y.cpu())[0,chan], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                used = np.zeros(contours_t.size)
                f_p = 0 # False Positive
                f_n = 0 # False Negative
                true = 0
                for i in range(contours_p.size): # distance between corners are less than sqrt(4^2+4^2)
                    found = 0
                    for k in range(contours_t.size):
                        if(     (abs(contours_t[k][0] - contours_p[i][0]) < math.sqrt(32)) and
                                (abs(contours_t[k][1] - contours_p[i][1]) < math.sqrt(32)) and
                                (abs(contours_t[k][2] - contours_p[i][2]) < math.sqrt(32)) and
                                (abs(contours_t[k][3] - contours_p[i][3]) < math.sqrt(32))):
                            found = k
                    if found == 0:
                        f_n+=1
                        total +=1
                    else:
                        used[found]=1
                f_p = np.count_nonzero(used == 0)
                acc[chan] = true/(true+f_n+f_p)

            # ENDED HERE
            # TODO: Check for errors, normal accuracy calculation, line 123:format

            acc += ((preds == y).sum().item())
            total += y.size(0)


        acc /= total
    return true_y, pred_y, acc


batch_size = 2
n_itr = 25
lr = 0.001

trainset = CudaVisionDataset(dir_path='./data/train')  # (image, target) set
testset = CudaVisionDataset(dir_path='./data/test')  # (image, target) set

train_split, valid_split = random_split(trainset, [int(len(trainset) * 0.8),
                                                   int(len(trainset) - (len(trainset) * 0.8))])

train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_split, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

model = NimbroNet(4, 4)
model = model.cuda()

# torch.set_default_tensor_type(torch.cuda.FloatTensor)
loss_type = "Mean Squared Error"
optim = torch.optim.Adam

model.train()
loss = nn.MSELoss()
optimizer = optim(model.parameters(), lr=lr)
loss_list, train_acc, valid_acc = train(train_dataloader, valid_dataloader,
                                        iters=n_itr, model_save_path="model1.pth")
t1 = time.time()

_, _, test_acc = acc(valid_dataloader)
print("Time to converge: {} sec".format(t1))
print("Best train accuracy={}, valid accuracy={} (based on the later)".
      format(train_acc, valid_acc))
print("Test accuracy on best model={}".format(test_acc))
