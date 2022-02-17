import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class CNNndp(nn.Module):
    def __init__(self,   N=5, state_index=np.arange(1), hidden_activation=F.relu
                 , b_init_value=0.1):
        '''
        Deep Neural Network for ndp
        :param N: No of basis functions (int)
        :param state_index: index of available states (np.array)
        :param hidden_activation: hidden layer activation function
        :param b_init_value: initial bias value
        '''

        super(CNNndp, self).__init__()
        self.hidden_activation = hidden_activation
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 200)
        self.b_init_value = b_init_value
        self.CNN_layers = [self.conv1, self.conv2, self.fc1]

        self.output_size = N * len(state_index) + 2 * len(state_index)
        self.layer_sizes = [784, 200, 100, 200, 2 * self.output_size, self.output_size]

        self.middle_layers = []
        for i in range(1, len(self.layer_sizes)-1):
            layer = nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1])
            fanin_init(layer.weight)
            layer.bias.data.fill_(b_init_value)
            self.middle_layers.append(layer)
            self.add_module("middle_layer_" + str(i), layer)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.last_fc = init_(nn.Linear(self.layer_sizes[-1], self.output_size))

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))

        for layer in self.middle_layers:
            x = F.relu(layer(x))

        output = self.last_fc(x) * 1000
        return output


    def reset_parameters(self):
        '''
        re-initialization of parameters
        '''
        # Cnn network
        for layer in self.CNN_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0.01)
            elif isinstance(layer, nn.Linear):
                layer.bias.data.fill_(0.01)
                nn.init.xavier_uniform_(layer.weight)


        # middle layer
        for layer in self.middle_layers:
            fanin_init(layer.weight)
            layer.bias.data.fill_(self.b_init_value)

        # Last layer
        self.last_fc = init(self.last_fc, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

