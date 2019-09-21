import json
import os

import numpy as np

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

class CudaVisionDataset(Dataset):

    def __init__(self, dir_path, no_of_classes=4,
                 channel_lut=None, blob_rad = 8):
        """
        :param dir_path:
        :param no_of_classes:
        :param channel_lut:
        :param blob_rad: Try to keep odd
        """
        super(CudaVisionDataset, self).__init__()
        self.img_paths, self.annot_paths = read_files(img_dir_path = dir_path)
        self.no_of_classes = no_of_classes
        if channel_lut is None:
            channel_lut = {'Head':0, 'Hand':1, 'Leg':2, 'Trunk':3}
        self.channel_lut = channel_lut
        self.blob_rad = blob_rad

    def __getitem__(self, index):
        #print(index)
        img = cv2.imread(self.img_paths[index], 1)
        # print(img.shape) # Shape is l x w x c
        l,w,c = img.shape
        img = cv2.resize(img, dsize=(480,640))
        # print(img.shape)
        # cv2.imshow('image', img)
        # cv2.waitKey()

        annot = parse_annotations(self.annot_paths[index])

        targets = np.zeros((self.no_of_classes, l, w), dtype='float')

        for k in annot.keys():
            # print(annot[k])
            for p in annot[k]:
                targets[self.channel_lut[k], int(p[0]), int(p[1])] = 1.0

        # print(targets.shape)
        targets = torch.Tensor(targets)
        downsampler = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        targets = downsampler(targets).numpy()
        # print(targets.shape)

        gaussian_2d = define_2d_gaussian(rad=self.blob_rad)
        target_coords_downsampled = np.where(targets==1)
        # print(np.where(targets==1))
        for i in range(target_coords_downsampled[0].shape[0]):
            targets[target_coords_downsampled[0][i]] = centre_and_place(targets[target_coords_downsampled[0][i]],
                                                    gaussian_2d, self.blob_rad, (target_coords_downsampled[1][i],
                                                    target_coords_downsampled[2][i]))

        img = np.moveaxis(img,2,0) # cv2 images are l X w X c

        img = torch.Tensor(img)
        targets = torch.Tensor(targets)

        return img, targets


    def __len__(self):
        return len(self.img_paths)


class CudavisionDataloader:

    def __call__(self, dir_path='./data/Images', batch_size=2):
        dataset = CudaVisionDataset(dir_path)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def centre_and_place(arr, g, rad, coords):

    lt = int(rad/2)
    rt = int(rad/2)+1
    if rad%2==0:
        rt -= 1
    # print(max(0,coords[0]-lt), min(arr.shape[0],coords[0]+rt),
    #     max(0,coords[1]-lt), min(arr.shape[1],coords[1]+rt))
    arr[max(0,coords[0]-lt): min(arr.shape[0],coords[0]+rt),
        max(0,coords[1]-lt): min(arr.shape[1],coords[1]+rt)] = g

    return arr

def define_2d_gaussian(rad=5, mu=0.0, sigma=8.0):

    x, y = np.meshgrid(np.linspace(-1, 1, rad), np.linspace(-1, 1, rad))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g

def parse_annotations(fname):
    with open(fname) as f:
        data = json.load(f)

    data = data['shapes']
    dpoints = {'Head':np.array([]), 'Hand': np.array([]), 'Leg': np.array([]), 'Trunk': np.array([])}
    for d in data:
        # print(d)
        label = d['label']
        bbox_coords = np.array(d['points'])
        mid = (bbox_coords[0]+bbox_coords[1])/2
        # print(bbox_coords, mid)
        dpoints[label] = np.append(dpoints[label], mid[::-1], axis=0)

    for k in dpoints.keys():
        dpoints[k] = dpoints[k].reshape((-1,2))
        # print(bbox_coords)
        # points
    # print(dpoints)
    return dpoints


def read_files(img_dir_path, img_format='.jpg', annot_format='.json', annot_folder_name='json'):
    img_paths = []
    annot_paths = []
    #img_dir_path += '/Images/'
    if os.path.isdir(img_dir_path):
        print("Folder exists. Reading..")

    for r, _, f in os.walk(img_dir_path + '/Images/'):
        for file in f:
            if img_format in file:
                img_paths.append(os.path.join(r, file))
    if len(img_paths) == 0:
        print("No Images in given path available. Check directory or format.")

    del_index = []
    for index, f in enumerate(img_paths):
        f_split = f.split("/")
        f_split[-1] = f_split[-1].replace(img_format, annot_format)
        f_split[-2] = annot_folder_name
        #print(f_split)
        annot_path = os.path.join(*f_split)

        if os.path.exists(annot_path):
            annot_paths.append(annot_path)
        else:
            print(annot_path)
            print("{} does not exist. Please verify.")
            exit(1)
            del_index.append(index)

    return img_paths, annot_paths

#
# if __name__ == "__main__":
#     # read_files('./data/Images')
#     # parse_annotations('./igus Humanoid Open Platform 331.json')
#
#     # dataset = CudaVisionDataset('/content/gdrive/My Drive/CUDA_Lab_Final_Project_Dataset/Albert_Saikat')
#     dataset = CudaVisionDataset('./data/')
#     for i in enumerate(dataset):
#         print(0)