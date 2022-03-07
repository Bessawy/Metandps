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

#UNKNOWN BUG
def regular_train_(loss_fn, eval_loss_fn, task_model, X_train, Y_train, exp_cfg):
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


def regular_train(loss_fn, eval_loss_fn, task_model, X_train, Y_train, exp_cfg):
    loss_trace = []
    lr = exp_cfg['inner_lr']
    inner_itr = exp_cfg['inner_itr']
    optimizer = torch.optim.Adam(task_model.parameters(), lr=lr )

    with higher.innerloop_ctx(task_model, optimizer,
                              copy_initial_weights=False) as (fmodel, diffopt):
        # update model parameters via meta loss
        for i in range(inner_itr):
            yp = fmodel(X_train, Y_train[:, 0, :])
            pred_loss = loss_fn.forward(yp, Y_train[:])
            diffopt.step(pred_loss)

        torch.save(fmodel.state_dict(), 'regular_train_weights.pth')

    return loss_trace


def eval(exp_cfg, train_loss_fn, eval_loss_fn, x, y, name):
    seed = exp_cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
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

        loss = regular_train(loss_fn=train_loss_fn, eval_loss_fn=eval_loss_fn, task_model = ndpn_ ,
                             X_train = x, Y_train =y, exp_cfg=exp_cfg)

        ndpn_.load_state_dict(torch.load("regular_train_weights.pth"))
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

def main(exp_cfg):
    # --------------load-data----------------------------------------------
    data_path = './dmp/data/s-mnist/40x40-smnist.mat'
    images, outputs, scale, or_tr = MatLoader.load_data(data_path, load_original_trajectories=True)
    images = np.array([cv2.resize(img, (28, 28)) for img in images]) / 255.0
    data_sep = Separate()
    digit_indices = data_sep.no_separation()
    input_size = images.shape[1] * images.shape[2]
    Zero_ind = digit_indices[2].astype(int)
    #One_ind = digit_indices[1].astype(int)
    #Three_ind = digit_indices[3][:100].astype(int)
    #Four_ind = digit_indices[4][:100].astype(int)
    #train_indx = np.concatenate((Zero_ind, One_ind))
    #test_indx = np.concatenate((Three_ind, Four_ind))
    inds = np.arange(1205)
    np.random.shuffle(inds)
    test_inds = inds[1200:]
    train_inds = inds[:1200]

    X = torch.Tensor(images[Zero_ind]).float()
    Y = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[Zero_ind]
    #X_new = torch.Tensor(images[test_indx]).float()
    #Y_new = torch.Tensor(np.array(or_tr)[:, :, :2]).float()[test_indx]

    time = str(datetime.now())
    time = time.replace(' ', '_')
    time = time.replace(':', '_')
    time = time.replace('-', '_')
    time = time.replace('.', '_')
    model_save_path = './data/' + '_' + time
    exp_cfg['model_save_path'] = model_save_path
    os.mkdir(model_save_path)

    k = 1
    T = 300 / k
    N = 30
    Y = Y[:, ::k, :]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    X_train = X[train_inds]
    Y_train = Y[train_inds]
    X_test = X[test_inds]
    Y_test = Y[test_inds]
    #X_new =  X_new.to(device)
    #Y_new = Y_new.to(device)
    Y_train = Y_train.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    num_epochs = exp_cfg['n_outer_iter']
    batch_size = exp_cfg['batch_size']
    inner_itr =  exp_cfg['inner_itr']
    outer_lr = exp_cfg['outer_lr']
    inner_lr = exp_cfg['inner_lr']

    # -------------------initialize policies and optimizers----------------------------#
    DNN = CNNndp(N=N, state_index=np.arange(2))
    ndpn = Ndp(DNN, T=T, l=1, N=N, state_index=np.arange(2))
    #torch.save(ndpn.state_dict(), 'initial_weights.pth')
    optimizer = torch.optim.Adam(ndpn.parameters(), lr=inner_lr)
    #
    meta_loss_network = ML3_smnist(301, [20, 20])
    #meta_loss_network = LearnedLossWeightedMse(2,2,301)
    meta_optimizer = torch.optim.SGD(meta_loss_network.parameters(), lr=outer_lr)
    #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(meta_optimizer, 0.9)
    #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(meta_optimizer, 'min', factor=0.8, patience=2)
    ndpn.to(torch.device("cuda:0"))
    meta_loss_network.to(torch.device("cuda:0"))

    MSE = torch.nn.MSELoss()
    MAE = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')

    losses = []
    x_axis = []
    training_losses = []
    test_losses_ml3 = []
    test_losses_reg = []
    test_losses_reg_var = []
    np.random.seed(0)
    torch.manual_seed(0)
    for outer_i in range(num_epochs):
        loss = []
        inds = np.arange(X_train.shape[0])
        np.random.shuffle(inds)

        for ind in np.split(inds, len(inds) // batch_size):
            ndpn.load_state_dict(torch.load('initial_weights.pth'))
            #ndpn.reset_parameters()

            meta_optimizer.zero_grad()
            with higher.innerloop_ctx(ndpn, optimizer,
                                      copy_initial_weights=False) as (fmodel, diffopt):
                # update model parameters via meta loss
                for i in range(inner_itr):
                    yp = fmodel(X_train[ind], Y_train[ind, 0, :])
                    pred_loss = meta_loss_network.forward(yp, Y_train[ind])
                    diffopt.step(pred_loss)

                yp = fmodel(X_train[ind], Y_train[ind, 0, :])
                task_loss = MAE(yp, Y_train[ind])
                task_loss.backward()
                loss.append(task_loss.item())

            meta_optimizer.step()

        avg_qry_loss = sum(loss)/len(loss)
        training_losses.append(avg_qry_loss)

        #lr_scheduler.step(avg_qry_loss)
        if outer_i % 1 == 0:

            print("Learning rate: ", meta_optimizer.param_groups[0]['lr'])

            res_test_eval_reg = eval(exp_cfg=exp_cfg,
                                     train_loss_fn=MAE, eval_loss_fn=MAE, x=X_test, y=Y_test, name = "norm")
            res_test_eval_ml3 = eval(exp_cfg=exp_cfg,
                                    train_loss_fn=meta_loss_network, eval_loss_fn=MAE, x=X_test, y=Y_test, name = "ml3")
            res_train_eval_ml3 = eval(exp_cfg=exp_cfg,
                                 train_loss_fn=meta_loss_network, eval_loss_fn=MAE, x=X_train[:5], y=Y_train[:5], name="ml3_train")

            x_axis.append(outer_i)
            test_loss_ml3 = np.mean(res_test_eval_ml3['mse'])
            test_loss_reg = np.mean(res_test_eval_reg['mse'])
            train_losses_ml3 = np.mean(res_train_eval_ml3['mse'])
            test_losses_ml3.append(test_loss_ml3)
            test_losses_reg.append(test_loss_reg)


            print(
                f'[Epoch {outer_i:.2f}] Train Loss: {avg_qry_loss:.2f}]| Test Loss ML3: {test_loss_ml3:.2f} | TestLoss REG: {test_loss_reg:.2f} '
                f'| TrainLoss ml3: {train_losses_ml3:.2f}'
            )

            #if(test_loss_ml3 < 2.4):
             #   break

    plt.plot(x_axis, test_losses_reg, label='ndp MAE loss')
    plt.plot(x_axis, test_losses_ml3, label='ndp ml3 loss')
    plt.legend()
    plt.savefig(model_save_path + "/train_result_{}.png".format(inner_itr))
    plt.clf()

    plt.plot(training_losses, label='training losses')
    plt.legend()
    plt.savefig(model_save_path + "/training_lossses.png")
    plt.clf()


    #np.save('meta_losses_NN_i50_o100.npy', np.array(losses))
    #torch.save(meta_loss_network.state_dict(), 'meta_loss_network_manyerror.pth')

if __name__ == "__main__":
    exp_cfg = {}
    exp_cfg['seed'] = 0
    exp_cfg['num_train_tasks'] = 1
    exp_cfg['num_test_tasks'] = 1
    exp_cfg['n_outer_iter'] = 100
    exp_cfg['n_gradient_steps_at_test'] = 100
    exp_cfg['batch_size'] = 100
    exp_cfg['inner_lr'] = 2.5e-3
    exp_cfg['outer_lr'] = 1e-3
    exp_cfg['inner_itr'] = 18

    main(exp_cfg)