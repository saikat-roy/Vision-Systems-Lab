import sys

sys.path.append("E:/Vision-Systems-Lab/Project9/")

from model import *
from utils import *
from train import *
#from google.colab.patches import cv2_imshow
import cv2

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

                drawing_t = x[0].cpu().numpy().astype('uint8')
                drawing_p = x[0].cpu().numpy().astype('uint8')
                drawing_t = np.moveaxis(drawing_t, 0, 2).copy()
                drawing_p = np.moveaxis(drawing_p, 0, 2).copy()
                # cv2.imshow('before_1',drawing_t)


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
                        cv2.circle(drawing_p, (int(centers_p[i][0] * 4) - 9, int(centers_p[i][1] * 4) - 13), int(6), (255, 152, 30), -1)

                    image, contours_t, _ = cv2.findContours(np.array((y.cpu())[0, chan] * 255).astype(np.uint8),
                                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours_poly = [None] * len(contours_t)
                    boundRect_t = [None] * len(contours_t)
                    centers_t = [None] * len(contours_t)
                    radius_t = [None] * len(contours_t)

                    for i, c in enumerate(contours_t):
                        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                        centers_t[i], radius_t[i] = cv2.minEnclosingCircle(contours_poly[i])

                    for i in range(len(boundRect_t)):
                        cv2.circle(drawing_t, (int(centers_t[i][0] * 4), int(centers_t[i][1] * 4)), int(6), (255, 0, 0), -1)

                    # drawing_t.convertTo(result8u,CV_8U);
                    cv2.imshow('1',drawing_t)
                    cv2.imshow('2',drawing_p)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    pass

model = Resnet18NimbroNet().cuda()
trainset = CudaVisionDataset(dir_path='./data/train')  # (image, target) set

train_split, valid_split, test_split = random_split(trainset, [300,52,100])
#
# train_dataloader = torch.utils.data.DataLoader(train_split, batch_size=batch_size, shuffle=True)
# valid_dataloader = torch.utils.data.DataLoader(valid_split, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_split, batch_size=2, shuffle=True)
model.load_state_dict(torch.load('model1 (3).pth'))
picture(test_dataloader)