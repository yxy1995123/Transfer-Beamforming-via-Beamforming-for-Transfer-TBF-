import scipy.io
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
import BatchLoader
import datetime
import random

import Net
def generate_txt(base_path):
    img_path = base_path + '/data/'
    IDList = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    random.shuffle(IDList)



    split = int(len(IDList) * 0.8)
    train = IDList[:split]
    test = IDList[split:]


    train_txt = base_path + 'train.txt'
    test_txt = base_path + 'test.txt'
    np.savetxt(train_txt, train, fmt='%s')
    np.savetxt(test_txt, test, fmt='%s')
if __name__ == '__main__':
    store_path = os.path.abspath(os.path.dirname(__file__)) + '/models'
    if not os.path.isdir(store_path):
        print('No folder named \"models/\"')
        exit()

    model_lst = [x for x in sorted(os.listdir(store_path)) if x.endswith('.pkl')]

    img_path = 'Z:\yangxueyuan_disk/beamtag\phase_dataset/'
    label_path = 'Z:\yangxueyuan_disk/beamtag\phase_dataset/'
    if 'train.txt' in os.listdir(img_path):
        print(' train.txt has been generated')
    else:
        generate_txt(img_path)
        print('train.txt not generated')
    epochs = 1000
    batchsize = 64

    data = BatchLoader.FileDataset(img_path)
    data = BatchLoader.BatchDataset(data, batchsize,  mode='eval')
    # data = BatchLoader.FileDataset(img_path, label_path)
    # data = BatchLoader.BatchDataset(data, batchsize, mode='eval')

    if len(model_lst) == 0:
        print('No previous model found, please check it')
        exit()
    else:
        print('Find previous model %s' % model_lst[-1])
        # model = resnet.resnet50(pretrained=False)
        # checkpoint = torch.load(store_path + '/%s' % model_lst[-1])
        # model.load_state_dict(checkpoint)
        model = Net.model()
        checkpoint = torch.load(store_path + '/%s' % model_lst[-1])
        model.load_state_dict(checkpoint)

        model.eval()

    error = []
    for i in range(data.num_of_patch):
        batch, info = data.EvalBatch()
        ID1 = str(int(info['ID'])).zfill(4)

        batch = Variable(torch.FloatTensor(batch), requires_grad=False)
        if torch.cuda.is_available():
            model.cuda()
            batch = batch.cuda()

        theta = model(batch)
        result = np.array(theta.squeeze().cpu().detach())


        # scipy.io.savemat('./results/result_' + ID1 + '.mat', {'result': result })


