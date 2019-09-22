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
    plt.ylim(0,1)
    p1 = []
    p2 = []
    for itr in range(iters):
        av_itr_loss = 0.0
        model.train()
        for batch_id, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            preds = model(x)

            #y = F.pad(y, (4, 5, 7, 6, 0, 0, 0, 0), mode='constant', value=0)
            batch_loss = loss(preds, y)  # ????
            batch_loss.backward()
            optimizer.step()
            av_itr_loss += (1 / y.size(0)) * batch_loss.item()
        loss_l.append(av_itr_loss)

        _, _, train_acc, _, _, _ = acc(train_dataloader, itr)
        _, _, valid_acc, t_p, f_p, f_n = acc(valid_dataloader, itr)

        if not suppress_output:
            if itr % 1 == 0 or itr == iters - 1:
                print("Epoch {}: Loss={}, Training Accuracy:{}, Validation Accuracy:{}"
                      .format(itr, av_itr_loss, train_acc, valid_acc))
                for chan in range(t_p.shape[0]):
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
        elif valid_acc < 0.65 *best_valid_acc:
            print("Model Acc. below threshold. Loading best model..")
            model.load_state_dict(torch.load(model_save_path))

        f_detection = 1 - (t_p / (t_p + f_p + 1))
        recall = t_p / (t_p + f_n + 1)
        p1 += [f_detection]
        p2 += [recall]

        plt.subplot(2, 1, 1)
        plt.plot(p1)
        plt.title('False Detection')

        plt.subplot(2, 1, 2)
        plt.plot(p2)
        plt.title('Recall')
        plt.pause(0.5)
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

            for b_id in range(y.size(0)):
                acc_chan = np.zeros(preds.shape[1])
#                 print(preds.shape)
                for chan in range(preds.shape[1]):

                    # Erosion
                    kernel = np.ones((3, 3), np.uint8)
                    (_, preds_thresh) = cv2.threshold(preds[b_id, chan], gray_thresh, 255, 0)
                    preds_erosion = cv2.erode(preds_thresh, kernel, iterations=1)

                    # Dilation
                    preds_dilation = cv2.dilate(preds_erosion, kernel, iterations=1)
#                     preds_dilation = preds_thresh
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

#                     if itr == 2:
#                         drawing = np.zeros((preds_dilation.astype(np.uint8).shape[0],
#                                             preds_dilation.astype(np.uint8).shape[1], 3), dtype=np.uint8)

#                         for i in range(len(boundRect_p)):
#                             color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
#                             cv2.drawContours(drawing, contours_poly, i, color)
#                             cv2.rectangle(drawing, (int(boundRect_p[i][0]), int(boundRect_p[i][1])),
#                                           (int(boundRect_p[i][0] + boundRect_p[i][2]),
#                                            int(boundRect_p[i][1] + boundRect_p[i][3])),
#                                           color, 2)

#                         cv2_imshow(drawing)
#                         #cv2.waitKey(0)
#                         # cv2.destroyAllWindows()

#                         cv2_imshow(y.cpu().numpy())
#                         #cv2.waitKey(0)
#                         # cv2.destroyAllWindows()

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