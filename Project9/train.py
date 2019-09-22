import sys

sys.path.append("E:/Vision-Systems-Lab/Project9/")

from model import *
from utils import *

import numpy as np
from torch.utils.data import dataloader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import time
import random

import matplotlib
import matplotlib.pyplot as plt


random.seed(12345)
np.set_printoptions(threshold=sys.maxsize)

def picture(dataloader):
    acc = 0.0
    true_y = []
    pred_y = []
    total = 0.0
    model.eval()

    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            if (batch_id == 1):
                x = x.cuda()
                y = y.cuda()

                drawing_t = x[0].cpu().numpy()
                drawing_p = x[0].cpu().numpy()

                drawing_t =np.moveaxis(drawing_t, 0, -1)
                drawing_p = np.moveaxis(drawing_p, 0, -1)

                for chan in range(4):
                    preds = np.array(model(x).cpu()[0][chan])
                    targets = np.array(y.cpu()[0][chan])

                    # (thresh, preds) = cv2.threshold(preds, 0.4, 255, 0)

                    kernel = np.ones((3, 3), np.uint8)
                    # Erosion

                    (_, preds_thresh) = cv2.threshold(preds, 0.4, 255, 0)
                    preds_erosion = cv2.erode(preds_thresh, kernel, iterations=1)

                    # Dilation
                    preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                    # Contour Detection

                    image, contours_p, _ = cv2.findContours((preds_dilation).astype(np.uint8), cv2.RETR_TREE,
                                                            cv2.CHAIN_APPROX_SIMPLE)
                    contours_poly = [None] * len(contours_p)
                    boundRect_p = [None] * len(contours_p)
                    centers_p = [None] * len(contours_p)
                    radius_p = [None] * len(contours_p)

                    for i, c in enumerate(contours_p):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        centers_p[i], radius_p[i] = cv2.minEnclosingCircle(contours_poly[i])

                    for i in range(len(boundRect_p)):
                        cv2.circle(drawing_p, (int(centers_p[i][0] * 4) - 9, int(centers_p[i][1] * 4) - 13), int(8),
                                   (255, 152, 30), 15)

                    a = np.array((y.cpu())[0, chan] * 255).astype(np.uint8)
                    image, contours_t, _ = cv2.findContours((targets*255).astype(np.uint8),
                                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_poly = [None] * len(contours_t)
                    boundRect_t = [None] * len(contours_t)
                    centers_t = [None] * len(contours_t)
                    radius_t = [None] * len(contours_t)

                    for i, c in enumerate(contours_t):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        centers_t[i], radius_t[i] = cv2.minEnclosingCircle(contours_poly[i])

                    for i in range(len(boundRect_t)):
                        cv2.circle(drawing_t, (int(centers_t[i][0] * 4), int(centers_t[i][1] * 4)), int(20), (255, 20, 20),
                                   15)

                print(np.min(drawing_t == drawing_p))
                cv2.imshow('image1',drawing_t/255)
                cv2.imshow('image2',drawing_p/255)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                pass

def train(train_dataloader, valid_dataloader, iters=20, suppress_output=False,
          gray_thresh=0, model_save_path="best.pth"):
    """
    Trains the model on the given dataloader and returns the loss per epoch
    """
    loss_l = []
    train_acc_l = []
    valid_acc_l = []
    best_valid_acc = 0.0
    equiv_train_acc = 0.0
    best_false_positive = 0.0
    best_true_positive = 0.0
    best_false_negative = 0.0

    plt.ion()
    plt.show()

    # First create some toy data:

    # Creates two subplots and unpacks the output array immediately
    f, (head_ax,hand_ax,leg_ax,trunk_ax, avg_ax) = plt.subplots(5,1)

    head_ax.set_title('Head')
    head_ax.set_ylim(0,1)
    hand_ax.set_title('Hand')
    hand_ax.set_ylim(0, 1)
    leg_ax.set_title('Leg')
    leg_ax.set_ylim(0, 1)
    trunk_ax.set_title('Trunk')
    trunk_ax.set_ylim(0, 1)
    avg_ax.set_title('Average')
    avg_ax.set_ylim(0, 1)

    for itr in range(iters):
        av_itr_loss = 0.0
        model.train()
        for batch_id, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            preds = model(x)

            batch_loss = loss(preds, y)  # ????
            batch_loss.backward()
            optimizer.step()
            av_itr_loss += (1 / y.size(0)) * batch_loss.item()
        loss_l.append(av_itr_loss)

        _, _, train_acc, _, _, _ = acc(train_dataloader, itr)
        _, _, valid_acc, t_p, f_p, f_n = acc(valid_dataloader, itr)

        if not suppress_output:
            if itr % 5 == 0 or itr == iters - 1:
                print("Epoch {}: Loss={}, Training Accuracy:{}, Validation Accuracy:{}"
                      .format(itr, av_itr_loss, train_acc, valid_acc))
                for chan in range(t_p.shape[0]):
                    #valid_dataloader.
                    print('Chan: ', chan, ' True: ', t_p[chan], '\t FP: ', f_p[chan], '\t FN: ', f_n[chan])
        train_acc_l.append(train_acc)
        valid_acc_l.append(valid_acc)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            equiv_train_acc = train_acc
            best_false_positive = np.average(f_p)
            best_true_positive = np.average(t_p)
            best_false_negative = np.average(f_n)
            valid_acc_l.append(valid_acc)
            torch.save(model.state_dict(), model_save_path)

        # f_detection = 1 - (t_p / (t_p + f_p + 1))
        # recall = t_p / (t_p + f_n + 1)
        # head += [f_detection[0], recall[0]]
        # p2 += [recall]
        #
        # head_ax.plot()
        # hand_ax.plot(f_detection[0], recall[0])
        # leg_ax.plot(f_detection[0], recall[0])
        # trunk_ax.plot(f_detection[0], recall[0])
        # avg_ax.plot(np.average(f_detection), np.average(recall[0]))
        #
        #
        #
        # plt.pause(0.5)
        #    model.load_state_dict(torch.load(model_save_path))

    return loss_l, equiv_train_acc, best_valid_acc, best_true_positive, best_false_positive, best_false_negative


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
    arr = np.array(x)
    for i in range(x.shape[0]):
        with open(name + '.txt', 'a') as the_file:
            the_file.write(np.array2string(arr[i]))
            the_file.write('\n')


def acc(dataloader, itr, tresh=4, gray_thresh=0.4):
    """
    Calculate accuracy of predictions from model for dataloader.
    :param gray_thresh:
    :param tresh:
    :param dataloader: dataloader to evaluate
    :return:
    """
    acc = 0.0
    true_y = []
    pred_y = []
    total = 0.0
    model.eval()
    f_p = np.zeros(4)  # False Positive
    f_n = np.zeros(4)  # False Negative
    true = np.zeros(4)
    with torch.no_grad():
        for batch_id, (x, y) in enumerate(dataloader):
            x = x.cuda()
            y = y.cuda()
            # y = F.pad(y, (4, 5, 7, 6, 0, 0, 0, 0), mode='constant', value=0)

            preds = model(x).cpu().numpy()

            for b_id in range(dataloader.batch_size):
                acc_chan = np.zeros(preds.shape[1])

                for chan in range(preds.shape[1]):

                    # Erosion
                    kernel = np.ones((3, 3), np.uint8)
                    (_, preds_thresh) = cv2.threshold(preds[b_id, chan], gray_thresh, 255, 0)
                    preds_erosion = cv2.erode(preds_thresh, kernel, iterations=1)

                    # Dilation
                    preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)

                    image, contours_p, _ = cv2.findContours(preds_dilation.astype(np.uint8), cv2.RETR_TREE,
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

                    # if itr == 20:
                    #     drawing = np.zeros((preds_dilation.astype(np.uint8).shape[0],
                    #                         preds_dilation.astype(np.uint8).shape[1], 3), dtype=np.uint8)
                    #
                    #     for i in range(len(boundRect_p)):
                    #         color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                    #         cv2.drawContours(drawing, contours_poly, i, color)
                    #         cv2.rectangle(drawing, (int(boundRect_p[i][0]), int(boundRect_p[i][1])),
                    #                       (int(boundRect_p[i][0] + boundRect_p[i][2]),
                    #                        int(boundRect_p[i][1] + boundRect_p[i][3])),
                    #                       color, 2)
                    #
                    #     cv2.imshow('image', drawing)
                    #     cv2.waitKey(0)
                    #     # cv2.destroyAllWindows()
                    #
                    #     cv2.imshow('image', y)
                    #     cv2.waitKey(0)
                    #     # cv2.destroyAllWindows()

                    used = np.zeros(len(boundRect_t))
                    for i in range(len(boundRect_p)):

                        found = -1

                        for k in range(len(boundRect_t)):
                            x_t = min(boundRect_t[k][0], boundRect_t[k][1]) + abs(
                                (boundRect_t[k][0] - boundRect_t[k][1])) / 2
                            y_t = min(boundRect_t[k][2], boundRect_t[k][3]) + abs(
                                (boundRect_t[k][2] - boundRect_t[k][3])) / 2

                            x_p = min(boundRect_p[i][0], boundRect_p[i][1]) + abs(
                                (boundRect_p[i][0] - boundRect_p[i][1])) / 2
                            y_p = min(boundRect_p[i][2], boundRect_p[i][3]) + abs(
                                (boundRect_p[i][2] - boundRect_p[i][3])) / 2

                            if (
                                    abs(x_t - x_p) < tresh and
                                    abs(y_t - y_p) < tresh):
                                found = k
                                true[chan] += 1
                                # break

                        if found == -1:
                            f_p[chan] += 1
                        else:
                            used[found] = 1
                    f_n[chan] += np.count_nonzero(used == 0)
                    # acc_chan[chan] = (true + 0.001) / ((true + f_n + f_p) + 0.001)

                # acc += acc_chan.sum() / acc_chan.size
                # total += 1

        acc = np.average(true) / (np.average(true) + np.average(f_n) + np.average(f_p))
    return true_y, pred_y, acc, true, f_p, f_n


if __name__ == "__main__":
    batch_size = 4
    n_itr = 100
    lr = 0.01

    trainset = CudaVisionDataset(dir_path='./data/train')  # (image, target) set

    train_split, valid_split, test_split = random_split(trainset, [300,52,100])

    train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_split, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_split, batch_size=batch_size, shuffle=True)

    model = Resnet18NimbroNet()
    model = model.cuda()

    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    loss_type = "Mean Squared Error"
    optim = torch.optim.Adam

    model.train()
    loss = nn.MSELoss()
    optimizer = optim(model.parameters(), lr=lr)

    loss_list = np.zeros(5)
    _, train_acc, valid_acc, t_p, f_p, f_n = train(train_dataloader, valid_dataloader,
                                                   iters=n_itr, model_save_path="model1.pth")

    t1 = time.time()

    # _, _, test_acc = acc(valid_dataloader)


    f_detection = 1 - (t_p / (t_p + f_p + 1))
    recall = t_p / (t_p + f_n + 1)

    print("Time to converge: {} sec".format(t1))
    print("Best train accuracy={}, valid accuracy={}, false detection={}, recall={} ".
          format(train_acc, valid_acc, f_detection, recall))

    # print("Test accuracy on best model={}".format(test_acc))



    picture(test_dataloader)
