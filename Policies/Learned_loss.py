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

        nn.init.uniform_(self.output.weight, a=0.0, b=1)

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


class StructuredLossMFRLNN(nn.Module):
    def __init__(self, out_dim, batch_size):
        super(StructuredLossMFRLNN, self).__init__()

        self.out_dim = out_dim
        self.batch_size = batch_size
        self.reset()

    def reset(self):
        self.phi = torch.nn.Parameter(torch.ones(self.out_dim))

    def forward(self, state, mean, sig, action):
        dists = torch.distributions.Normal(mean, sig)
        logprob = dists.log_prob(action)
        rew_1 = (torch.sum(state, axis=1))
        rew_2 = (torch.sum(action, axis=1))


        # https://discuss.pytorch.org/t/repeat-a-nn-parameter-for-efficient-computation/25659/2
        # should hold and share gradients
        phi = self.phi.repeat(self.batch_size).view(-1, 1)
        phi2 = self.phi2.repeat(self.batch_size).view(-1, 1)

        l = (phi * rew_1 + phi2 * rew_2)
        selected_logprobs = l * logprob.sum(dim=-1)
        return -selected_logprobs.mean()


class DeepNNrewardslearning(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super(DeepNNrewardslearning, self).__init__()

        self.in_dim = in_dim
        self.activation = nn.ELU()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 1)

    def forward(self, state, mean, sig, actions, rewards):
        y = torch.cat((state, mean, sig), dim=1)
        y = self.activation(self.fc1(y))
        y = self.activation(self.fc2(y))
        y = self.output(y)*1e1

        dists = torch.distributions.Normal(mean, sig)
        logprob = dists.log_prob(actions)
        selected_logprobs = y * logprob.sum(dim=-1)
        return -selected_logprobs.mean()

class PPORewardsLearning(nn.Module):
    def __init__(self, hidden=64):
        super(PPORewardsLearning, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 10, 5, 1)
        self.activation = nn.ReLU()
        self.fc1 = nn.Linear(10*4*4 , hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, 10)



    def forward(self, image):
        image = image.view(-1, 1, 28, 28)

        y = F.relu(self.conv1(image))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2, 2)

        y = y.view(-1, 4 * 4 * 10)

        x = self.fc1(y)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        reward = self.output(x)

        return reward.view(-1, 2, 5)


