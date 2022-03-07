# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def fanin_init(tensor):
    '''
    :param tensor: torch.tensor
    used to initialize network parameters
    '''
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)



def init(module, weight_init, bias_init, gain=1):
    '''
      :param tensor: torch.tensor
      used to initialize network parameters
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            module.bias.data.fill_(0.001)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class ML3_smnist(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(ML3_smnist, self).__init__()
        self.hidden_activation = F.relu
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.B1 = nn.BatchNorm1d(hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.B2 = nn.BatchNorm1d(hidden_dim[1])
        self.output = nn.Linear(hidden_dim[1], 1)
        self.last = nn.Softplus()
        self.middle_layers = [self.fc2]
        self.reset()

    def forward(self, y_in, y_target):
        '''
        :param y_in: shope(batch, 301, 2)
        :param y_target: shope(batch, 301, 2)
        :return: loss
        '''
        y = torch.abs(y_in - y_target)
        y = torch.sum(y, dim=2)

        y = self.hidden_activation(self.fc1(y))
        y = self.hidden_activation(self.fc2(y))
        y = self.last(self.output(y))
        return y.mean()

    def reset(self):
        for m in self.modules():
            weight_init(m)

        nn.init.uniform_(self.output.weight, a=0.0, b=0.05)



class LearnedLossWeightedMse(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LearnedLossWeightedMse, self).__init__()
        self.hidden_activation = F.relu
        self.fc1 = nn.Linear(301, 20)
        self.fc2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 301)
        self.out_dim = out_dim
        self.reset()

    def forward(self, y_in, y_target):
        err = torch.abs(y_in - y_target)
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
        err = (y_in - y_target)
        return self.phi[0] * err.mean()

    def reset(self):
        self.phi = torch.nn.Parameter(torch.ones(self.out_dim)*2.0)


    def get_parameters(self):
        return self.phi