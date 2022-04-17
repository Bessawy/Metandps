import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
from dmp.utils.dmp_layer import DMPIntegrator
from dmp.utils.dmp_layer import DMPParameters

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

class CNNndpPolicy(nn.Module):
    def __init__(self, state_space=2, action_space=2,
                 N=3, T=5, l=1, tau=1,
                 state_index=np.arange(2), b_init_value=0.1,
                 rbf='gaussian', az=True,
                 only_g=False):
        '''
        Deep Neural Network for ndp
        :param N: No of basis functions (int)
        :param state_index: index of available states (np.array)
        :param hidden_activation: hidden layer activation function
        :param b_init_value: initial bias value
        '''

        super(CNNndpPolicy, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.T, self.N, self.l = T, N, l
        self.hidden = 64
        dt = 1.0 / (T * self.l)
        self.output_size = N * len(state_index) + len(state_index)

        self.state_index = state_index
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, a_z=az)
        self.func = DMPIntegrator(rbf=rbf, only_g=only_g, az=az)
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        self.fc1 = torch.nn.Linear(state_space + 1, self.hidden)
        fanin_init(self.fc1.weight)
        self.fc1.bias.data.fill_(b_init_value)

        # actor
        self.fc2_mean = torch.nn.Linear(self.hidden, self.output_size)
        fanin_init(self.fc2_mean.weight)
        self.fc2_mean.bias.data.fill_(b_init_value)
        self.sigma = torch.nn.Linear(self.hidden, action_space)

        # critic
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = init(self.fc2_value, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

    def forward(self, state):
        x = state.view(-1, 3)

        x = self.fc1(x)
        x = torch.tanh(x) * 1e-1

        y0 = state[:, :2]
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01

        # critic for T actions
        value = self.fc2_value(x).repeat(1, self.T).view(-1, self.T, 2)
        value = torch.transpose(value, 1, 2)
        # sigma for T actions
        sigma = torch.sigmoid(self.sigma(x))*1.0 + 0.001
        sigma = sigma.view(-1, 2, 1)

        # actions

        ndp_wg = self.fc2_mean(x)
        y, dy, ddy = self.func.forward(ndp_wg, self.DMPp, self.param_grad, None, y0, dy0)  # y = [200,301]
        y = y.view(state.shape[0], len(self.state_index), -1)
        y = y[:, :, ::self.l]
        a = y[:, :, 1:] - y[:, :, :-1]

        return a, sigma, value
