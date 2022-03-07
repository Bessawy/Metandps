import torch
import numpy as np
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class sminst_Data_Sampler(object):
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.list_inds = []
        self.inds = np.arange(X.shape[0])

    def shuffle_data(self):
        np.random.shuffle(self.inds)
        for ind in np.split(self.inds, len(self.inds) // self.batch_size):
            self.list_inds.append(ind)
            return ind

    def sample(self):
        if len(self.list_inds) == 0:
            self.shuffle_data()
        ind = self.list_inds.pop(-1)
        return self.X[ind], self.Y[ind], self.Y[ind, 0, :]

    def get_data(self):
        return self.X[:100], self.Y[:100]