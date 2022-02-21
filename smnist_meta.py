import numpy as np
import torch
import cv2
from datetime import datetime
import os
import sys
from Ndp import Ndp
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from Policies.CNN_ndp import CNNndp
from dmp.utils.smnist_loader import MatLoader, Separate
from Policies.Learned_loss import ML3_smnist
from ml3_.ml3_train import meta_train_il
from ml3_.ml3_test import meta_test_il
from dmp.utils.utils import sminst_Data_Sampler


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

def meta_config_0():
    exp_cfg = {}
    exp_cfg['seed'] = 0
    exp_cfg['batch_size'] = 100
    exp_cfg['num_train_tasks'] = 1
    exp_cfg['num_test_tasks'] = 1
    exp_cfg['n_outer_iter'] = 500
    exp_cfg['n_gradient_steps_at_test'] = 100
    exp_cfg['inner_lr'] = 1e-3
    exp_cfg['outer_lr'] = 1e-3
    exp_cfg['data_path'] = './dmp/data/s-mnist/40x40-smnist.mat'
    exp_cfg['n_inner_iter'] = 10
    time = str(datetime.now())
    time = time.replace(' ', '_')
    time = time.replace(':', '_')
    time = time.replace('-', '_')
    time = time.replace('.', '_')
    exp_cfg['model_save_path'] = './data/' + '_' + time
    return exp_cfg


if __name__ == '__main__':
    # -------------------Initialization----------------------------#
    exp_cfg = meta_config_0()
    data_path = exp_cfg['data_path']
    seed = exp_cfg['seed']
    inner_lr = exp_cfg['inner_lr']
    outer_lr = exp_cfg['outer_lr']
    np.random.seed(seed)

    model_save_path = exp_cfg['model_save_path']
    num_epochs = exp_cfg['n_outer_iter']
    batch_size = exp_cfg['batch_size']
    k = 1
    T = 300 / k
    N = 30

    # -------------------Loading Data set----------------------------#
    images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
    Sep = Separate()
    Sep_images_inds = Sep.no_separation(images)

    images = np.array([cv2.resize(img, (28, 28)) for img in images]) / 255.0
    input_size = images.shape[1] * images.shape[2]

    inds = np.arange(12000)
    np.random.shuffle(inds)

    test_inds = inds[10000:]
    train_inds = inds[:10000]

    X = torch.Tensor(images[:12000]).float()
    Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]

    Y = Y[:, ::k, :]

    X_train = X[train_inds]
    Y_train = Y[train_inds]
    X_test = X[test_inds]
    Y_test = Y[test_inds]

    # -------------------initialize policies and optimizers----------------------------#
    Meta_learn_Network = ML3_smnist(301*2, [100, 100])
    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    task_ndps = [ndpn]
    torch.save(ndpn.state_dict(), 'init_only.pth')

    # -------------------Meta Training----------------------------#
    meta_objective = torch.nn.MSELoss()

    if 1:
        tasks_to_train = sminst_Data_Sampler(X_train, Y_train, batch_size)
        meta_train_il(Meta_learn_Network, meta_objective, tasks_to_train, exp_cfg,
                task_ndps, count_parameters)
    else:
        exp_folder = exp_cfg['model_save_path']
        os.mkdir(model_save_path)
        meta_test_il(ndpn, Meta_learn_Network, 1000, batch_size, X_train[:100], Y_train[:100],
                 X_test[:100], Y_test[:100], model_save_path)



