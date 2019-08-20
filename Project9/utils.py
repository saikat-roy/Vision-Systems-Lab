import json
import os

import numpy as np

import cv2
from torch.utils.data import Dataset, DataLoader


class CudaVisionDataset(Dataset):

    def __init__(self, dir_path):

        super(CudaVisionDataset, self).__init__()
        self.img_paths, self.annot_paths = read_files(img_dir_path = dir_path)

    def __getitem__(self, index):
        #print(index)
        img = cv2.imread(self.img_paths[index],1)
        print(img.shape) # Shape is l x w x c
        cv2.imshow('image', img)
        cv2.waitKey()

        annot = parse_annotations(self.annot_paths[index])
        exit()
        return img, annot


    def __len__(self):
        return self.len


def parse_annotations(fname):
    with open(fname) as f:
        data = json.load(f)

    data = data['shapes']
    dpoints = {'Head':np.array([]), 'Hand': np.array([]), 'Leg': np.array([]), 'Trunk': np.array([])}
    for d in data:
        print(d)
        label = d['label']
        bbox_coords = np.array(d['points'])
        mid = (bbox_coords[0]+bbox_coords[1])/2
        print(bbox_coords, mid)
        dpoints[label] = np.append(dpoints[label], mid, axis=0)

    for k in dpoints.keys():
        dpoints[k] = dpoints[k].reshape((-1,2))
        # print(bbox_coords)
        # points
    print(dpoints)


def read_files(img_dir_path, img_format='.jpg', annot_format='.json', annot_folder_name='json'):
    img_paths = []
    annot_paths = []
    if os.path.isdir(img_dir_path):
        print("Folder exists. Reading..")

    for r, _, f in os.walk(img_dir_path):
        for file in f:
            if img_format in file:
                img_paths.append(os.path.join(r, file))
    if len(img_paths) == 0:
        print("No Images in given path available. Check directory or format.")

    del_index = []
    for index, f in enumerate(img_paths):
        f_split = f.split("/")
        f_split[-1] = f_split[-1].replace(img_format, annot_format)
        f_split[2] = annot_folder_name
        # print(f_split)
        annot_path = os.path.join(*f_split)

        if os.path.exists(annot_path):
            annot_paths.append(annot_path)
        else:
            print("{} does not exist. Please verify.".format())
            exit(1)
            del_index.append(index)

    return img_paths, annot_paths


if __name__ == "__main__":
    # read_files('./data/Images')
    # parse_annotations('./igus Humanoid Open Platform 331.json')

    dataset = CudaVisionDataset('./data/Images')
    for i in enumerate(dataset):
        0
