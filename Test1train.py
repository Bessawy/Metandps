
from dmp.utils.smnist_loader import MatLoader, Separate
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
from Policies.Learned_loss import ML3_smnist, LearnedLossWeightedMse, SimpleLoss

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
    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    ndpn.load_state_dict(torch.load('initial_weights_ndp_1demo.pth'))
    meta_loss_network = SimpleLoss()
    #meta_loss_network = ML3_smnist(301 * 2, [50, 50])
    #meta_loss_network = LearnedLossWeightedMse(1, 1, 301)
    meta_loss_network.load_state_dict(torch.load('meta_loss_network_1demo.pth'))
    optimizer = torch.optim.Adam(ndpn.parameters(), lr=learning_rate)

    count_parameters(ndpn)
    loss_history = []

    for epoch in range(num_epochs):
        for ind in range(1):
            optimizer.zero_grad()
            y_h = ndpn.forward(X_train, Y_train[ind, 0, :])  # y is a 2D pose for all batches
            loss = meta_loss_network.forward(y_h, Y_train)
            loss_history.append(torch.mean((y_h - Y_train) ** 2).item())
            loss.backward()
            optimizer.step()

        print('Epoch: ' + str(epoch) + ', Loss ' + str(loss_history[-1]))
        #torch.save(ndpn.state_dict(), model_save_path + '/model.pt')



    np.save('meta_test_losses_ndp.npy', np.array(loss_history))
    plt.plot(loss_history)
    plt.show