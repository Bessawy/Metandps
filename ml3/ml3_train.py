import os

import numpy as np
import torch.nn as nn
import torch
import higher


def meta_train_il(meta_loss_Network, meta_optimizer, meta_objective, tasks_to_train, tasks_to_test, exp_cfg,
               task_ndps, ndp_opts):
    '''
    meta_train function for supervised learning
    :param meta_loss_Network: Learned Loss NN
    :param meta_optimizer: Optimizer for the meta_loss_network
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
    exp_folder = exp_cfg['exp_folder']

    for outer_i in range(n_outer_iter):
        x_spt, y_spt, x_qry, y_qry, y0 = tasks_to_train.sample()  # TO-DO

        for i in range(num_tasks):
            task_ndps[i].reset_parameters()

        training_losses = []
        for _ in range(n_inner_iter):

            meta_optimizer.zero_grad()

            for i in range(num_tasks):
                # zero gradients wrt to meta loss parameters
                with higher.innerloop_ctx(task_ndps[i], ndp_opts[i],
                                          copy_initial_weights=False) as (fmodel, diffopt):

                    # update model parameters via meta loss
                    yh = fmodel(x_spt[i], y0)
                    pred_loss = meta_loss_Network.forward(yh, y_spt[i]) # TO_DO
                    diffopt.step(pred_loss)

                    # compute task loss with new model
                    yp = fmodel(x_spt[i], y0)
                    task_loss = meta_objective(yp, y_spt[i])

                    # this accumulates gradients wrt to meta parameters
                    task_loss.backward()
                    training_losses.append(task_loss.item())# TO-DO (visualize training losses)

        meta_optimizer.step()
        torch.save(meta_optimizer.state_dict(), f'{exp_folder}/ml3_loss_reacher.pt')