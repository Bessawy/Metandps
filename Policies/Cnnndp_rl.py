import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import sys
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

class CNNndprl(nn.Module):
    def __init__(self, state_space=2, action_space=2,
                 N=3, T=5, l=1, tau=1,
                 state_index=np.arange(1), b_init_value=0.1):
        '''
        Deep Neural Network for ndp
        :param N: No of basis functions (int)
        :param state_index: index of available states (np.array)
        :param hidden_activation: hidden layer activation function
        :param b_init_value: initial bias value
        '''

        super(CNNndprl, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.T, self.N, self.l = T, N, l
        self.hidden = 64
        dt = 1.0 / (T * self.l)
        self.output_size = N * len(state_index) + len(state_index)

        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None)
        self.func = DMPIntegrator()
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 10, 5, 1)

        self.fc1 = torch.nn.Linear(state_space + 1 + 10 * 4 * 4, self.hidden)
        fanin_init(self.fc1.weight)
        self.fc1.bias.data.fill_(b_init_value)

        feature_size = 10*4*4 + 3

        #actor
        self.fc2_mean = torch.nn.Linear(self.hidden,  self.output_size)
        fanin_init(self.fc2_mean.weight)
        self.fc2_mean.bias.data.fill_(b_init_value)
        self.sigma = torch.nn.Linear(self.hidden, action_space)

        #critic
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = init(self.fc2_value, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

    def forward(self, x, image):
        image = image.view(-1, 1, 28, 28)
        x = x.view(-1, 3)

        y = F.relu(self.conv1(image))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2, 2)

        y = y.view(-1, 4 * 4 * 10)
        x = torch.cat((x, y), dim=1)

        x = self.fc1(x)
        x = torch.tanh(x)

        y0 = x[:, :2]
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01

        print(y0.reshape)

        #actor
        ndp_wg = self.fc2_mean(x)
        y, dy, ddy = self.func.forward(ndp_wg, self.DMPp, self.param_grad, None, y0, dy0)  #
        y = y.view(input.shape[0], len(self.state_index), -1)
        y = y[:, :, ::self.l]
        a = y[:, :, 1:] - y[:, :, :-1]
        print(a.shape)
        sys.exit()

        # critic
        value = self.fc2_value(x).repeat(1, self.T)




        return a, value


c