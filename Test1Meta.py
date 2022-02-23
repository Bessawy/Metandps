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

if __name__ == '__main__':
    data_path = './dmp/data/s-mnist/40x40-smnist.mat'
    images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
    images = np.array([cv2.resize(img, (28, 28)) for img in images]) / 255.0
    input_size = images.shape[1] * images.shape[2]

    inds = np.arange(1)
    np.random.shuffle(inds)
    test_inds = inds[:1]
    train_inds = inds[:1]
    X = torch.Tensor(images[:1]).float()
    Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:1]

    time = str(datetime.now())
    time = time.replace(' ', '_')
    time = time.replace(':', '_')
    time = time.replace('-', '_')
    time = time.replace('.', '_')
    model_save_path = './data/' + '_' + time
    os.mkdir(model_save_path)
    k = 1
    T = 300 / k
    N = 30
    learning_rate = 1e-3
    Y = Y[:, ::k, :]

    X_train = X[train_inds]
    Y_train = Y[train_inds]
    X_test = X[test_inds]
    Y_test = Y[test_inds]

    num_epochs = 500
    batch_size = 100

    # -------------------initialize policies and optimizers----------------------------#
    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    optimizer = torch.optim.Adam(ndpn.parameters(), lr=1e-3)
    #meta_loss_network = ML3_smnist(301*2, [10, 10]) first improved
    #meta_loss_network = ML3_smnist(301 * 2, [50, 50])
    #meta_loss_network = LearnedLossWeightedMse(1,1,301)
    meta_loss_network=SimpleLoss()
    meta_optimizer = torch.optim.Adam(meta_loss_network.parameters(), lr=1e-3)
    count_parameters(meta_loss_network)
    ndpn.load_state_dict(torch.load('initial_weights_ndp_1demo.pth'))
    try:
        for outer_i in range(100):
            #print(f'[Epoch {outer_i:.2f}] loss: ', meta_loss_network.get_parameters())
            # Sample a batch of support and query images and labels.
            # -------------------------------------------------------
            # change between reset and reset_parameters
            # task_ndps[0].reset()
            inds = np.arange(1)
            np.random.shuffle(inds)
            test_inds = inds[:1]
            train_inds = inds[:1]
            X = torch.Tensor(images[:1]).float()
            Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:1]

            losses = []
            meta_optimizer.zero_grad()

            # zero gradients wrt to meta loss parameters
            with higher.innerloop_ctx(ndpn, optimizer,
                                          copy_initial_weights=False) as (fmodel, diffopt):

                # update model parameters via meta loss
                for _ in range(10):
                    yp = fmodel(X_train, Y_train[0, 0, :])  # fmodel remain the same
                    pred_loss = meta_loss_network.forward(yp, Y_train)
                    diffopt.step(pred_loss)


                yp = fmodel(X_train, Y_train[0, 0, :])
                task_loss = torch.mean((yp - Y_train) ** 2)

                task_loss.backward()
                losses.append(task_loss.item())

            meta_optimizer.step()

            print(f'[Epoch {outer_i:.2f}] loss: {task_loss.item():.2f}]')


        np.save('meta_losses.npy', np.array(losses))
        torch.save(meta_loss_network.state_dict(), 'meta_loss_network_1demo.pth')
    except:
        np.save('meta_losses.npy', np.array(losses))
