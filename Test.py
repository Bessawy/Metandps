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
from Policies.Learned_loss import ML3_smnist, LearnedLossWeightedMse, SimpleLoss
import higher
from ml3_.ml3_train import meta_train_il
from ml3_.ml3_test import meta_test_il
from dmp.utils.utils import sminst_Data_Sampler
def loss_func(y_target, y_predict):
    loss = torch.abs(y_target - y_predict) ** 2
    loss = torch.sum(loss, dim=2) + 1e-16
    loss = loss.sqrt()
    return loss.mean()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def regular_train(loss_fn, eval_loss_fn, task_model, X_train, Y_train, exp_cfg):
    loss_trace = []
    lr = exp_cfg['inner_lr']
    inner_itr = exp_cfg['inner_itr']
    optimizer = torch.optim.Adam(task_model.parameters(), lr=lr )


    for i in range(inner_itr):
        optimizer.zero_grad()
        y_pred = task_model(X_train, Y_train[:, 0, :])

        loss = loss_fn(y_pred, Y_train)

        loss.backward()
        optimizer.step()

        eval_loss = eval_loss_fn(y_pred, Y_train)
        loss_trace.append(eval_loss.item())

    return loss_trace

def eval(exp_cfg, train_loss_fn, eval_loss_fn, x, y, name):
    seed = exp_cfg['seed']
    mse = []
    nmse = []
    loss_trace = []

    k = 1
    T = 300 / k
    N = 30

    for i in range(1):


        DNN_ = CNNndp(N=N, state_index=np.arange(2))
        ndpn_ = Ndp(DNN_, T=T, l=1, N=N, state_index=np.arange(2))
        ndpn_.load_state_dict(torch.load('initial_weights.pth'))
        ndpn_.to(torch.device("cuda:0"))

        yp = ndpn_.forward(x, y[:, 0, :])

        loss = regular_train(loss_fn=train_loss_fn, eval_loss_fn=eval_loss_fn, task_model = ndpn_ ,
                             X_train = x, Y_train =y, exp_cfg=exp_cfg)

        yp = ndpn_.forward(x, y[:, 0, :])
        l = eval_loss_fn(yp, y)

        for k in range(yp.shape[0]):
            plt.plot(0.667 * yp[k, :, 0].detach().cpu().numpy(), -0.667 * yp[k, :, 1].detach().cpu().numpy(),
                     c='r', linewidth=5)
            plt.axis('off')
            plt.savefig(exp_cfg['model_save_path'] + '/train_img_' + name + '.png')
            plt.clf()

        mse.append(l.item())
        nmse.append(l.item()/y.cpu().numpy().var())
        loss_trace.append(loss)

    res = {'nmse': nmse, 'mse': mse, 'loss_trace': loss_trace}
    return res

# --------------load-data----------------------------------------------
data_path = './dmp/data/s-mnist/40x40-smnist.mat'
images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
images = np.array([cv2.resize(img, (28, 28)) for img in images]) / 255.0
data_sep = Separate()
digit_indices = data_sep.no_separation()
input_size = images.shape[1] * images.shape[2]
Zero_ind = digit_indices[2].astype(int)

inds = np.arange(105)
np.random.shuffle(inds)
test_inds = inds[100:]
train_inds = inds[:100]

X = torch.Tensor(images[Zero_ind]).float()
Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[Zero_ind]

k = 1
T = 300 / k
N = 30
Y = Y[:, ::k, :]

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = X[train_inds]
Y_train = Y[train_inds]
X_test = X[test_inds]
Y_test = Y[test_inds]
#X_new =  X_new.to(device)
#Y_new = Y_new.to(device)
#Y_train = Y_train.to(device)
#X_train = X_train.to(device)
#X_test = X_test.to(device)
#Y_test = Y_test.to(device)



# -------------------initialize policies and optimizers----------------------------#
DNN = CNNndp(N=N, state_index=np.arange(2))
ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
optimizer = torch.optim.Adam(ndpn.parameters(), lr=1e-3)
meta_loss_network = ML3_smnist(301, [20, 20])
meta_optimizer = torch.optim.SGD(meta_loss_network.parameters(), lr=1e-3)




ndpn.load_state_dict(torch.load('initial_weights.pth'))

with higher.innerloop_ctx(ndpn, optimizer,
                              copy_initial_weights=False) as (fmodel, diffopt):

    for i in range(10):
        yp = fmodel.forward(X_train[:5], Y_train[:5, 0, :])
        pred_loss = (yp.mean() - 10.0)**2
        print("prediction loss: ", pred_loss)
        diffopt.step(pred_loss)



print("------------------------------------------------------------------------------")
for i in range(10):
    optimizer.zero_grad()
    y_pred = ndpn.forward(X_train[:5], Y_train[:5, 0, :])
    loss = (y_pred.mean() - 10.0)**2
    print("prediction loss: ", loss)
    loss.backward()
    optimizer.step()

