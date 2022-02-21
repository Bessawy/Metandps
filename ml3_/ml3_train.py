import os
import sys

import numpy as np
import torch.nn as nn
import torch
import higher


def meta_train_il(meta_loss_Network, meta_objective, data_train, exp_cfg,
               task_ndps, count_parameters, tasks_to_test = None):
    '''
    meta_train function for supervised learning
    :param meta_loss_Network: Learned Loss NN
    :param meta_objective: Task specific loss to optimize the meta_model
    :param tasks_to_train: (object) used to obtain data for training
    :param tasks_to_test: (object) used to sample data for testing
    :param exp_cfg: experiment parameters
    :param task_models: List of neural dynamic policies
    :param task_opts: optimizers of the neural dynamic policies
    :return: None
    '''

    num_tasks = exp_cfg['num_train_tasks']
    n_outer_iter = exp_cfg['n_outer_iter']
    n_inner_iter = exp_cfg['n_inner_iter']
    inner_lr = exp_cfg['inner_lr']
    outer_lr = exp_cfg['outer_lr']
    exp_folder = exp_cfg['model_save_path']
    batch_size = exp_cfg['batch_size']


    task_opts = []
    meta_optimizer = torch.optim.Adam(meta_loss_Network.parameters(), lr=1e-3)

    print("------Meta loss Network Parameters-----")
    count_parameters(meta_loss_Network)
    for i in range(len(task_ndps)):
        task_opts.append(torch.optim.SGD(task_ndps[i].parameters(), lr=inner_lr))
        print("------Ndps {} Network Parameters-----".format(i))
        count_parameters(task_ndps[i])



    for outer_i in range(1000):
        # Sample a batch of support and query images and labels.
        #-------------------------------------------------------
        # change between reset and reset_parameters
        # task_ndps[0].reset()
        task_ndps[0].reset()

        qry_losses = []

        X, Y= data_train.get_data()
        inds = np.arange(100)
        np.random.shuffle(inds)
        test_inds = inds[:100]
        train_inds = inds[:100]
        X_train = X[train_inds]
        Y_train = Y[train_inds]

        inds = np.arange(X_train.shape[0])
        np.random.shuffle(inds)



        for i_ in range(10):

            for ind in np.split(inds, len(inds) // batch_size):

                pred_losses = []
                meta_optimizer.zero_grad()
                per_iter = []

                for i in range(num_tasks):
                    # zero gradients wrt to meta loss parameters
                    with higher.innerloop_ctx(task_ndps[0], task_opts[0],
                                              copy_initial_weights=False) as (fmodel, diffopt):

                        # update model parameters via meta loss
                        yp = fmodel(X_train[ind], Y_train[ind, 0, :])  #fmodel remain the same
                        pred_loss = meta_loss_Network(yp, Y_train[ind])
                        print("predicted loss always zero: ", pred_loss.item())
                        diffopt.step(pred_loss)

                        # compute task loss with new model
                        yp = fmodel(X_train[ind], Y_train[ind, 0, :])
                        task_loss = torch.mean((yp - Y_train[ind]) ** 2)

                        task_loss.backward()
                        qry_losses.append(task_loss.item())


                    meta_optimizer.step()



        avg_qry_loss = sum(qry_losses)/len(qry_losses)

        if outer_i % 20 == 0:
            print(f'[Epoch {outer_i:.2f}] Last Train Loss: {task_loss:.2f}]')
            print(
                f'[Epoch {outer_i:.2f}] Train Loss: {avg_qry_loss:.2f}]'
            )
        torch.save(meta_optimizer.state_dict(), 'ml3_loss_sminst.pt')




'''
    for i in range(num_tasks):

        for outer_i in range(n_outer_iter):
            print("iter of training: ", outer_i)
            x_train, y_train, y0 = data_train.sample()  # TO-DO
            task_ndps[i].reset_parameters()
            training_losses = []

            for _ in range(n_inner_iter):
                meta_optimizer.zero_grad()
                # zero gradients wrt to meta loss parameters
                with higher.innerloop_ctx(task_ndps[i], ndp_opts[i],
                                          copy_initial_weights=False) as (fmodel, diffopt):

                    # update model parameters via meta loss
                    yh = fmodel(x_train, y0)
                    pred_loss = meta_loss_Network.forward(yh, y_train) # TO_DO
                    diffopt.step(pred_loss)

                    # compute task loss with new model
                    yp = fmodel(x_train, y0)
                    task_loss = meta_objective(yp, y_train)

                    # this accumulates gradients wrt to meta parameters
                    task_loss.backward()
                    training_losses.append(task_loss.item()) # TO-DO (visualize training losses)

                avg_training_loss = sum(training_losses) / n_inner_iter
                meta_optimizer.step()
                torch.save(meta_optimizer.state_dict(), f'{exp_folder}/ml3_loss_sminst.pt')
'''