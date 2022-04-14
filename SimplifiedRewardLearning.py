import matplotlib.pyplot as plt
from dmp.utils.smnist_loader import MatLoader, Separate
import multiprocessing
import multiprocessing.connection
from smnistenv import SMNIST
import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
from dmp.utils.dmp_layer import DMPIntegrator
from dmp.utils.dmp_layer import DMPParameters
from smnistenv import Worker
import higher
from utils import *
from Policies.optimizee import *
from Policies.Learned_loss import *
import logging
from visdom import Visdom

def rollout(env, model, device, batch_size=64):
    batch_rewards = []
    batch_actions = []
    batch_means_grad = []
    batch_sigs_grad = []
    batch_states = []
    total_rewards = []

    for _ in range(batch_size):
        states, means, sigs, actions, rewards, goals = [], [], [], [], [], []
        means_grad, sigs_grad = [], []
        s_0 = env.reset()
        g_0 = env.goal
        done = False

        while done == False:
            input_ = g_0 - s_0
            input_torch = obs_to_torch(input_, device).view(-1, 2)
            mean, sig = model(input_torch)

            dists = torch.distributions.Normal(mean, sig)
            action = dists.sample().cpu().numpy().squeeze()
            s_1, r, done, _ = env.step(action)

            # hold grad
            means_grad.append(mean)
            sigs_grad.append(sig)

            # Holds no_grads
            states.append(input_)
            actions.append(action)
            rewards.append(r)
            s_0 = s_1
            g_0 = env.goal

        batch_means_grad.extend(means_grad)
        batch_sigs_grad.extend(sigs_grad)
        batch_rewards.extend(discount_rewards(rewards))
        batch_states.extend(states)
        batch_actions.extend(actions)
        total_rewards.append(sum(rewards))

    rewards_torch = torch.FloatTensor(np.array(batch_rewards)).to(device)
    actions_torch = torch.FloatTensor(np.array(batch_actions)).to(device)
    states_torch = torch.FloatTensor(np.array(batch_states)).to(device)
    means_grad_torch = torch.cat(batch_means_grad, dim=0)
    sigs_grad_torch = torch.cat(batch_sigs_grad, dim=0)

    samples_nograd = {'states':states_torch, 'rewards':rewards_torch, 'actions':actions_torch}
    samples_grad = {'means':means_grad_torch, 'sigs':sigs_grad_torch}

    return samples_nograd, samples_grad, total_rewards

def regular_train(loss_fn, task_model, exp_cfg, learned):
    gamma = exp_cfg['gamma']
    seed = exp_cfg['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_iter = exp_cfg['n_inner_iter']
    lr = exp_cfg['inner_lr']
    env = exp_cfg['env']
    batch_size = exp_cfg['batch_size']
    device = exp_cfg['device']

    reward_trace = []
    loss_trace = []

    optimizer = torch.optim.SGD(task_model.parameters(), lr=lr)
    for ep in range(n_iter):
        optimizer.zero_grad()

        samples_nograd, samples_grad, total_rewards = rollout(env, task_model, device, batch_size)

        states, actions = samples_nograd['states'], samples_nograd['actions']
        rewards = samples_nograd['rewards']
        means, sigs = samples_grad['means'], samples_grad['sigs']

        reward_trace.append(np.mean(total_rewards))

        if not learned:
            loss = loss_fn(task_model, states, actions, rewards)
        else:
            loss = loss_fn(states, means, sigs, actions, rewards)

        loss.backward()
        optimizer.step()
        loss_trace.append(loss.item())
    return loss_trace, reward_trace

def meta_objective(model, states, actions, rewards):
   means, sigs = model(states)
   dists = torch.distributions.Normal(means, sigs)
   logprob = dists.log_prob(actions)
   selected_logprobs = rewards * logprob.sum(dim=-1)
   return -selected_logprobs.mean()

def train_metaloss(exp_cfg):
    torch.manual_seed(exp_cfg['seed'])
    np.random.seed(exp_cfg['seed'])
    env = exp_cfg['env']
    device = exp_cfg['device']
    ep_length = exp_cfg['ep_length']
    batch_size = exp_cfg['batch_size']

    policy = NNPolicy().to(device)
    policy_optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
    metaloss_network = DeepNNrewardslearning(6).to(device)
    metaoptimizer = torch.optim.Adam(metaloss_network.parameters(), lr=1e-3)
    torch.save(policy.state_dict(), 'policy.mdl')

    for outer_i in range(exp_cfg['n_outer_iter']):
        metaoptimizer.zero_grad()
        policy.load_state_dict(torch.load('policy.mdl'))

        with higher.innerloop_ctx(policy, policy_optimizer,
                                  copy_initial_weights=False) as (fmodel, diffopt):


            for i in range(exp_cfg['n_inner_iter']):
                samples_nograd, samples_grad, total_rewards = rollout(env, fmodel, device, batch_size)

                states, actions = samples_nograd['states'], samples_nograd['actions']
                rewards = samples_nograd['rewards']
                means, sigs = samples_grad['means'],samples_grad['sigs']

                pred_loss = metaloss_network(states, means, sigs, actions, rewards)
                diffopt.step(pred_loss)

            samples_nograd, samples_grad, total_rewards = rollout(env, fmodel, device, batch_size)

            states, actions = samples_nograd['states'], samples_nograd['actions']
            rewards = samples_nograd['rewards']

            Task_loss = meta_objective(fmodel, states, actions, rewards)
            Task_loss.backward()

        metaoptimizer.step()

        if outer_i % 20 == 0:
            policy.load_state_dict(torch.load('policy.mdl'))
            loss_trace_learned, reward_learned = regular_train(metaloss_network, policy, exp_cfg, True)
            policy.load_state_dict(torch.load('policy.mdl'))
            loss_trace_Reinf, reward_Reinf = regular_train(meta_objective, policy, exp_cfg, False)

            LOG.info(
         f' [Epoch {outer_i:.2f}] ML3 Reward: {reward_learned[-1]:.2f}]| : Reinforce Reward {reward_Reinf[-1]:.2f}'
            )

            if exp_cfg['plotting']:
                Y = np.stack([reward_learned, reward_Reinf]).transpose()
                X = np.stack([range(0, len(reward_Reinf), 1), range(0, len(reward_Reinf), 1)]).transpose()

                viz.line(Y=Y, X=X, env="rewards_ml3_reinf", win='eval_rewards',
                         opts=dict(showlegend=True, title='ml3/reinforce eval rewards',
                                   legend=['learned rewards', 'Standard rewards']))

                viz.line([[reward_learned[-1], reward_Reinf[-1]]], [outer_i], env="rewards_ml3_reinf",
                         win='rewards_update_steps', update='append',
                         opts=dict(showlegend=True, title='Final rewards at n_outer_iters',
                                   legend=['learned rewards', 'Standard rewards']))

def main():
    plotting = True
    global viz
    viz = Visdom()
    env_name = 'reacher_task_loss_learning_'

    if viz.check_connection():
        plotting = True
    else:
        plotting = False

    global LOG
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)


    exp_cfg = {}
    exp_cfg['gamma'] = 0.99
    exp_cfg['seed'] = 0
    exp_cfg['n_outer_iter'] = 100
    exp_cfg['n_inner_iter'] = 10
    exp_cfg['inner_lr'] = 1e-3
    exp_cfg['batch_size'] = 1
    exp_cfg['device'] = set_device()
    exp_cfg['digit'] = 2
    exp_cfg['ep_length'] = 300
    exp_cfg['env'] = SMNIST(digit_idx=exp_cfg['digit'], max_ep_length=exp_cfg['ep_length'])
    exp_cfg['plotting'] = plotting

    train_metaloss(exp_cfg)

if __name__ == "__main__":
    main()
