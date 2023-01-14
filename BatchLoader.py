import numpy as np
import torch
import torch.utils.data
import random
import os
import cv2
import scipy.io

class FileDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.img_path = path + 'data/'
        self.label_path =  self.img_path
        IDList =  np.loadtxt(path + '/train.txt',delimiter=',')
        # random.shuffle(IDList)
        # IDList = []
        # for i in IDList1:
        #     data = i.split('.')[0]
        #     IDList.append(data.zfill(4))
        self.IDList = IDList

    def __getitem__(self, index):
        tmp = {}
        # with open(self.label_path + '/%s.txt' % self.IDList[index], 'r') as f:
        #     buf = []
        #     for line in f:
        #         # line = line[:-1].split(' ')
        #         line = line[:-1].split(',')
        #         for i in range(1, len(line)):
        #             line[i] = float(line[i])
        #         Location = [line[0], line[1], line[2]]
        #         buf.append({'Location': Location})
        name = self.img_path+ str(int(self.IDList[index])).zfill(4) + '.mat'
        buf = []
        data = scipy.io.loadmat(name)
        img = data['bartlett'][0]
        buf.append({'Location': img})
        tmp['ID'] = self.IDList[index]
        tmp['Label'] = buf


        return tmp


    def GetImage(self, idx):
        name =  self.img_path+  str(int(self.IDList[idx])).zfill(4) + '.mat'
        data = scipy.io.loadmat(name)
        img = data['bartlett'][0]
        return img

    def __len__(self):
        return len(self.IDList)

class BatchDataset:
    def __init__(self, imgDataset, batchSize=1, mode='train'):
        self.imgDataset = imgDataset
        self.batchSize = batchSize

        self.mode = mode
        self.imgID = None


        self.info = self.getBatchInfo()
        self.Total = len(self.info)
        self.train_number = int(self.Total * 0.8)
        if mode == 'train':
            self.idx = 0
            self.num_of_patch = int(self.Total * 0.8)
        else:
            self.idx = self.train_number
            self.num_of_patch = int(self.Total * 0.2)
        # print len(self.info)
        # print self.info

    def getBatchInfo(self):

        data = []
        total = len(self.imgDataset)
        for idx, one in enumerate(self.imgDataset):
            ID = one['ID']
            # img = one['Image']
            allLabel = one['Label']
            for label in allLabel:
                data.append({'ID': ID,
                             'Index': idx,
                    'Location': label['Location']
                })
        return data

    def Next(self):
        batch = np.zeros([self.batchSize, 1, 1, 180], np.float)
        record = None
        for one in range(self.batchSize):
            data = self.info[self.idx]
            imgID = data['Index']
            if imgID != record:
                img = self.imgDataset.GetImage(imgID)


            batch[one, 0, 0, :] = 10 * img[ :]

            if self.mode == 'train':
                if self.idx + 1 < self.num_of_patch:
                    self.idx += 1
                else:
                    self.idx = 0
            else:
                if self.idx + 1 < self.Total:
                    self.idx += 1
                else:
                    self.idx = self.train_number
            gt = batch
        return batch, gt

    def EvalBatch(self):
        batch = np.zeros([1, 1, 1, 180], np.float)
        info = self.info[self.idx]
        imgID = info['Index']
        if imgID != self.imgID:
            self.img = self.imgDataset.GetImage(imgID)
            self.imgID = imgID
            batch[0, 0, 0, :] = 10 * self.img[:]
        if self.mode == 'train':
            if self.idx + 1 < self.num_of_patch:
                self.idx += 1
            else:
                self.idx = 0
        else:
            if self.idx + 1 < self.Total:
                self.idx += 1
            else:
                self.idx = self.train_number
        return batch,  info
