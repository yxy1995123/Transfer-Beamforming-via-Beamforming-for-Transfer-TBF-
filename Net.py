import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class SimpleNet(nn.Module):
    def __init__(self, in_dim ):
        super(SimpleNet,self).__init__()
        # n_hidden_1 = 256
        # n_hidden_2 = 512
        # n_hidden_3 = 1024
        in_dim = 180
        self.layer1 = nn.Sequential(nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(256, 8))
        self.layer7 = nn.Sequential(nn.Linear(8, 256),  nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer8 = nn.Sequential(nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        self.layer9 = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(True))
        self.layer10 = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(True))
        self.layer11 = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(True))
        self.layer12 = nn.Sequential(nn.Linear(256, 180))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x



def model():
    model = SimpleNet(180)
    return model
