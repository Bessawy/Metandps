import numpy as np
import torch.nn as nn
import torch
from dmp.utilits.dmp_layer import DMPIntegrator
from dmp.utilits.dmp_layer import DMPParameters


class Ndp(nn.Module):
    def __init__(self,  DNN, state_index=np.arange(1), N=5, T=10, l=10, tau=1):
        super().__init__()

        '''
        :DNN: deep neural network (Object) 
        :state_index: index of available states (np,array)
        :N: no of radial basis functions (int)
        :T: ???
        :l: ???
        :tau: time constant (int) 
        '''
        self.DNN = DNN

        self.N = N
        self.l = l
        self.T = T
        self.state_index = state_index

        self.output_size = N * len(state_index) + 2 * len(state_index)   # Output layer of DNN (w, g)
        dt = 1.0 / (T * self.l)

        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None)
        self.func = DMPIntegrator()
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)


    def forward(self, input, y0, return_preactivations=False):
        '''
        :param input: state input to the ndp
        :param y0: initial state
        :return: trajectory to follow
        '''
        output = self.DNN.forward(input)
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)  # y = [200,301]
        y = y.view(input.shape[0], len(self.state_index), -1)
        return y.transpose(2, 1)


    def reset_parameters(self):
        '''
           re-initialization of parameters
        '''
        try:
            self.DNN.reset_parameters()
        except:
            print("Policy must have function: reset_parameters()")

