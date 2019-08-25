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
np.set_printoptions(threshold=sys.maxsize)


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
            batch_loss = loss(preds, y)  # ????
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


def write(x, name):
    arr = np.array((x))
    for i in range(x.shape[0]):
        with open(name + '.txt', 'a') as the_file:
            the_file.write(np.array2string(arr[i]))
            the_file.write('\n')



def acc(dataloader, tresh=20):
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
    f_p = 0  # False Positive
    f_n = 0  # False Negative
    true = 0


    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            y = F.pad(y, (4, 5, 7, 6, 0, 0, 0, 0), mode='constant', value=0)

            preds = model(x).cpu().numpy()

            # preds = torch.argmax(preds, dim=1)  # ??
            # For each channel
            acc_chan = np.zeros(preds.shape[1])
            for chan in range(preds.shape[1]):
                # write(preds.cpu()[0, chan], 'preds')
                # write(y.cpu()[0, chan], 'truth')
                # Erosion
                kernel = np.ones((3, 3), np.uint8)
                (_, preds_thresh) = cv2.threshold(preds[0, chan], 0.4, 255, 0)
                preds_erosion = cv2.erode(preds_thresh, kernel, iterations=1)

                # Dilation
                preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                """                                                                                 ???
                After thresholding each
                output channel, we apply morphological erosion and dilation
                to eliminate negligible responses. Finally, we compute the
                object center coordinates from the achieved contours.
                """

                # Contour Detection
                #                 print(np.unique(preds_dilation))

                # preds_dilation = np.expand_dims(preds_dilation, axis=2)
                # write(preds_dilation, 'preds_d')
                image, contours_p, _ = cv2.findContours((preds_dilation).astype(np.uint8), cv2.RETR_TREE,
                                                        cv2.CHAIN_APPROX_SIMPLE)
                contours_poly = [None] * len(contours_p)
                boundRect_p = [None] * len(contours_p)
                for i, c in enumerate(contours_p):
                    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                    boundRect_p[i] = cv2.boundingRect(contours_poly[i])

                image, contours_t, _ = cv2.findContours(np.array((y.cpu())[0, chan] * 255).astype(np.uint8),
                                                        cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_poly = [None] * len(contours_t)
                boundRect_t = [None] * len(contours_t)
                for i, c in enumerate(contours_t):
                    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                    boundRect_t[i] = cv2.boundingRect(contours_poly[i])

                b = False
                if b:
                    drawing = np.zeros(((preds_dilation).astype(np.uint8).shape[0],
                                        (preds_dilation).astype(np.uint8).shape[1], 3), dtype=np.uint8)

                    for i in range(len(boundRect_p)):
                        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                        cv2.drawContours(drawing, contours_poly, i, color)
                        cv2.rectangle(drawing, (int(boundRect_p[i][0]), int(boundRect_p[i][1])), \
                                     (int(boundRect_p[i][0] + boundRect_p[i][2]), int(boundRect_p[i][1] + boundRect_p[i][3])),
                                     color, 2)

                    cv2.imshow('image', drawing)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    cv2.imshow('image', y)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()


                used = np.zeros(len(boundRect_t))

                for i in range(len(boundRect_p)):  # distance between corners are less than sqrt(4^2+4^2)
                    found = -1

                    for k in range(len(boundRect_t)):
                        if (
                                (abs((boundRect_t[k][0] - boundRect_p[i][0])) < tresh) and
                                (abs((boundRect_t[k][1] - boundRect_p[i][1])) < tresh) and
                                (abs((boundRect_t[k][2] - boundRect_p[i][2])) < tresh) and
                                (abs((boundRect_t[k][3] - boundRect_p[i][3])) < tresh)):
                            found = k
                            true += 1

                    if found == -1:
                        f_p += 1
                        #total += 1
                    else:
                        used[found] = 1
                f_n += np.count_nonzero(used == 0)
                #acc_chan[chan] = (true + 0.001) / ((true + f_n + f_p) + 0.001)

            #acc += acc_chan.sum() / acc_chan.size
            #total += 1

        acc = true/(true+f_n+f_p)
        print('True: ',true, '\t FP: ',f_p,'\t FN: ',f_n)
    return true_y, pred_y, acc


def calc(dataloader):
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
            y = y.cuda().cpu()
            y = F.pad(y, (4, 5, 7, 6, 0, 0, 0, 0), mode='constant', value=0)
            for b in range(2):
                for i in range(4):
                    preds = np.array(model(x).cpu()[b][i])
                    targets = np.array(y[b][i])
                    # preds = np.max(np.array((preds)), axis=1)

                    (thresh, preds) = cv2.threshold(preds, 0.4, 255, 0)

                    kernel = np.ones((3, 3), np.uint8)
                    preds_erosion = cv2.erode(preds, kernel, iterations=1)

                    # Dilation
                    preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                    cv2.imshow('image', preds)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    cv2.imshow('image', preds_dilation)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

                    cv2.imshow('image', targets)
                    cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            pass


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
# t1 = time.time()
#
# _, _, test_acc = acc(valid_dataloader)
# print("Time to converge: {} sec".format(t1))
# print("Best train accuracy={}, valid accuracy={} (based on the later)".
#       format(train_acc, valid_acc))
# print("Test accuracy on best model={}".format(test_acc))
