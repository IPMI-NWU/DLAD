import glob
import random
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils


class TrainDataset(Dataset):
    def __init__(self, image_dir, train_list_dir, transform):
        self.image_dir = image_dir
        self.train_list_dir = train_list_dir
        self.transform = transform

        self.image_files = []
        self.image_name = []
        with open(self.train_list_dir) as file:
            for item in file.readlines():
                image_name = item.strip("\n")
                self.image_name.append(image_name)
                self.image_files.append(os.path.join(self.image_dir, "{0}.png".format(image_name)))
                
    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)

        data = self.transform(Image.fromarray(image))
        label = torch.tensor(0, dtype=torch.float32)
        
        image_name = self.image_name[index % len(self.image_name)]

        return data, label, image_name

    def __len__(self):
        return len(self.image_files)

class TestDataset(Dataset):
    def __init__(self, normal_image_dir, normal_list_dir, abnormal_image_dir, abnormal_list_dir, transform):
        self.normal_image_dir = normal_image_dir
        self.abnormal_image_dir = abnormal_image_dir
        self.normal_list_dir = normal_list_dir
        self.abnormal_list_dir = abnormal_list_dir
        self.transform = transform
        self.image_name = []

        self.normal_image_files = []
        with open(self.normal_list_dir) as file:
            for item in file.readlines():
                image_name = item.strip("\n")
                self.image_name.append(image_name)
                self.normal_image_files.append(os.path.join(self.normal_image_dir, "{0}.png".format(image_name)))
        self.normal_labels = torch.zeros([len(self.normal_image_files)], dtype=torch.float32)

        self.abnormal_image_files = []
        with open(self.abnormal_list_dir) as file:
            for item in file.readlines():
                image_name = item.strip("\n")
                self.image_name.append(image_name)
                self.abnormal_image_files.append(os.path.join(self.abnormal_image_dir, "{0}.png".format(image_name)))
        self.abnormal_labels = torch.ones([len(self.abnormal_image_files)], dtype=torch.float32)

        self.image_files = []
        self.image_files.extend(self.normal_image_files)
        self.image_files.extend(self.abnormal_image_files)

        self.labels = []
        self.labels.extend(self.normal_labels)
        self.labels.extend(self.abnormal_labels)
                
    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)
        
        data = self.transform(Image.fromarray(image))
        label = self.labels[index % len(self.labels)]
        image_name = self.image_name[index % len(self.image_name)]

        return data, label, image_name

    def __len__(self):
        return len(self.image_files)

class TrainInpaintDataset(Dataset):
    def __init__(self, image_dir, train_list_dir, transform):
        self.image_dir = image_dir
        self.train_list_dir = train_list_dir
        self.transform = transform

        self.image_files = []
        self.image_name = []
        with open(self.train_list_dir) as file:
            for item in file.readlines():
                image_name = item.strip("\n")
                self.image_name.append(image_name)
                self.image_files.append(os.path.join(self.image_dir, "{0}.png".format(image_name)))
                
    def __getitem__(self, index):
        ori_image = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)

        ori_image = self.transform(Image.fromarray(ori_image))
        GM = GenMask(ori_image.size(1))
        upd_image = GM.getMask(ori_image)
        label = torch.tensor(0, dtype=torch.float32)

        image_name = self.image_name[index % len(self.image_name)]

        return upd_image, ori_image, label, image_name

    def __len__(self):
        return len(self.image_files)

class GenMask:
    def __init__(self, size):
        self.size = size

    def getMask(self, image):
        h = self.size
        w = self.size

        mask = np.ones((h, w), np.float32)

        n_holes = random.randint(1, 10) 

        for n in range(n_holes):
            y= np.random.randint(20, h-20)
            x = np.random.randint(20, w-20)
            length = np.random.randint(30, 80)
            width = np.random.randint(30, 80)

            y1 = np.clip(y-length//2, 0, h)
            y2 = np.clip(y+length//2, 0, h)
            x1 = np.clip(x-width//2, 0, w)
            x2 = np.clip(x+width//2, 0, w)

            mask[y1:y2, x1:x2]=0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask

        return image

class TestInpaintDataset(Dataset):
    def __init__(self, normal_image_dir, normal_list_dir, abnormal_image_dir, abnormal_list_dir, transform):
        self.normal_image_dir = normal_image_dir
        self.abnormal_image_dir = abnormal_image_dir
        self.normal_list_dir = normal_list_dir
        self.abnormal_list_dir = abnormal_list_dir
        self.transform = transform
        self.image_name = []

        self.normal_image_files = []
        with open(self.normal_list_dir) as file:
            for item in file.readlines():
                image_name = item.strip("\n")
                self.image_name.append(image_name)
                self.normal_image_files.append(os.path.join(self.normal_image_dir, "{0}.png".format(image_name)))
        self.normal_labels = torch.zeros([len(self.normal_image_files)], dtype=torch.float32)

        self.abnormal_image_files = []
        with open(self.abnormal_list_dir) as file:
            for item in file.readlines():
                image_name = item.strip("\n")
                self.image_name.append(image_name)
                self.abnormal_image_files.append(os.path.join(self.abnormal_image_dir, "{0}.png".format(image_name)))
        self.abnormal_labels = torch.ones([len(self.abnormal_image_files)], dtype=torch.float32)

        self.image_files = []
        self.image_files.extend(self.normal_image_files)
        self.image_files.extend(self.abnormal_image_files)

        self.labels = []
        self.labels.extend(self.normal_labels)
        self.labels.extend(self.abnormal_labels)
                
    def __getitem__(self, index):
        ori_image = cv2.imread(self.image_files[index % len(self.image_files)], cv2.IMREAD_GRAYSCALE)
        
        ori_image = self.transform(Image.fromarray(ori_image))
        GM = GenMask(ori_image.size(1))
        upd_image = GM.getMask(ori_image)
        label = self.labels[index % len(self.labels)]
        image_name = self.image_name[index % len(self.image_name)]

        return upd_image, ori_image,  label, image_name

    def __len__(self):
        return len(self.image_files)
