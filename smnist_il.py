
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

    inds = np.arange(12000)
    np.random.shuffle(inds)
    test_inds = inds[10000:]
    train_inds = inds[:10000]
    X = torch.Tensor(images[:12000]).float()
    Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[:12000]

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
    optimizer = torch.optim.Adam(ndpn.parameters(), lr=learning_rate)

    count_parameters(ndpn)



    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        inds = np.arange(X_train.shape[0])
        np.random.shuffle(inds)

        for ind in np.split(inds, len(inds) // batch_size):
            optimizer.zero_grad()

            y_h = ndpn.forward(X_train[ind], Y_train[ind, 0, :])  # y is a 2D pose for all batches

            #print(y_h.shape) 301 2D poses
            loss = torch.mean((y_h - Y_train[ind]) ** 2)
            print("Loss: ", loss)
            loss.backward()
            optimizer.step()

        print("save")
        torch.save(ndpn.state_dict(), model_save_path + '/model.pt')



        if epoch % 20 == 0:

            x_test = X_test[np.arange(100)]
            y_test = Y_test[np.arange(100)]
            y_htest = ndpn.forward(x_test, y_test[:, 0, :])
            for j in range(18):
                plt.figure(figsize=(8, 8))
                plt.plot(0.667 * y_h[j, :, 0].detach().cpu().numpy(), -0.667 * y_h[j, :, 1].detach().cpu().numpy(),
                         c='r', linewidth=5)
                plt.axis('off')
                plt.savefig(model_save_path + '/train_img_' + str(j) + '.png')

                plt.figure(figsize=(8, 8))
                img = X_train[ind][j].cpu().numpy() * 255
                img = np.asarray(img * 255, dtype=np.uint8)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(model_save_path + '/ground_train_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

                plt.figure(figsize=(8, 8))
                plt.plot(0.667 * y_htest[j, :, 0].detach().cpu().numpy(),
                         -0.667 * y_htest[j, :, 1].detach().cpu().numpy(), c='r', linewidth=5)
                plt.axis('off')
                plt.savefig(model_save_path + '/test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)

                plt.figure(figsize=(8, 8))
                img = X_test[j].cpu().numpy() * 255
                img = np.asarray(img * 255, dtype=np.uint8)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.savefig(model_save_path + '/ground_test_img_' + str(j) + '.png', bbox_inches='tight', pad_inches=0)
            test = ((y_htest - y_test) ** 2).mean(1).mean(1)
            print('Epoch: ' + str(epoch) + ', Test Error: ' + str(test.mean().item()))
