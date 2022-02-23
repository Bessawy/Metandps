# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            module.bias.data.fill_(0.01)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class ML3_smnist(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ML3_smnist, self).__init__()

        self.hidden_activation = F.relu
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.output = nn.Linear(hidden_dim[1], 1)
        self.outer_activation = nn.Softplus()
        self.reset()

    def forward(self, y_in, y_target):
        '''
        :param y_in: shope(batch, 301, 2)
        :param y_target: shope(batch, 301, 2)
        :return: loss
        '''

        #y = torch.cat((y_in, y_target), dim=1)
        y = torch.abs(y_in - y_target)**2
        y = y.reshape(y.shape[0], -1)
        y = self.hidden_activation(self.fc1(y))
        y = self.hidden_activation(self.fc2(y))
        y = self.output(y)
        y = torch.abs(y)
        return y.mean()

    def reset(self):
        for m in self.modules():
            weight_init(m)

class LearnedLossWeightedMse(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LearnedLossWeightedMse, self).__init__()

        self.out_dim = out_dim
        self.reset()

    def forward(self, y_in, y_target):
        err = ((y_in - y_target))**2
        l = torch.zeros_like(err[:,0])

        for i in range(self.out_dim):
            l+= self.phi[i] * err[:,i]

        return l.mean()

    def reset(self):
        self.phi = torch.nn.Parameter(torch.ones(self.out_dim))

    def get_parameters(self):
        return self.phi


class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.out_dim = 1
        self.reset()

    def forward(self, y_in, y_target):
        err = (y_in - y_target)**2
        return self.phi[0] * err.mean()

    def reset(self):
        self.phi = torch.nn.Parameter(torch.ones(self.out_dim)*2.0)


    def get_parameters(self):
        return self.phi