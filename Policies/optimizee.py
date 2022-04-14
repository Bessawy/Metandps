import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F

def weight_init_he(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Parameter):
        nn.init.normal_(module)

class Trjectory(torch.nn.Module):
    def __init__(self, n_actions=30, action_space=2, start_values=0.5):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_actions, action_space)* start_values)

    def forward(self, x):
        t = x[:,2:]
        t = t.type(torch.long).flatten()
        mean = self.weights[t,:]
        sigma = torch.tensor([1.0]).to(x.device)
        return mean, sigma

class NNPolicy(torch.nn.Module):
    def __init__(self, state_space=2, action_space=2):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.min_action, self.max_action = -5, 5

        self.hidden = 128
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, 2*action_space)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1, 2)
        x = self.fc1(x)
        x = torch.tanh(x)*1e-2

        action_mean = self.fc2_mean(x)
        mean, sigma = action_mean[:, :2], torch.exp(action_mean[:, 2:])
        #sigma = torch.sigmoid(sigma)*2.0 + 1e-4
        return mean, sigma

