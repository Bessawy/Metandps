import matplotlib.pyplot as plt
import multiprocessing
import multiprocessing.connection
import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
from smnistenv import Worker
from smnistenv import SMNIST
from utils import *
from Policies.optimizee import *
from Policies.Learned_loss import *
import logging
from visdom import Visdom
import higher





class Agent:
    def __init__(self, seed=0):
        # -----------------Loops--------------------
        self.n_outer_itr = 500
        self.n_inner_itr = 5
        self.epochs = 8
        # -----------------Workers------------------
        self.n_workers = 8
        self.worker_steps = 10
        # ---------------Advantage------------------
        self.lambdas = 0.96
        self.gamma = 0.99
        # ------------------NDP---------------------
        self.T = 5
        self.N = 3
        # -------------NeuralNetwork----------------
        self.clip = 0.1
        self.outer_lr = 1e-3
        self.inner_lr = 1e-3
        self.device = set_device()
        # -----------------Testing------------------
        self.seed = seed
        self.plotting = True



    def close_env(self):
        for worker in self.workers:
            worker.child.send(("close", None))

    def reset_env(self):
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        for worker in self.workers:
            worker.child.send(("timestep", None))
        for i, worker in enumerate(self.workers):
            self.timestep[i] = worker.child.recv()

    def initalize_env(self, desired_idx=2):
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        self.obs = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.timestep = np.zeros((self.n_workers, 1), dtype=np.float32)
        self.target_goal = np.zeros((self.n_workers, 28, 28))


    def rollout(self, model):
        '''
        :param model: Policy used to rollout
        :return: dict (no_grad)
        '''
        values = np.zeros((self.n_workers, self.worker_steps + 1, 2, self.T), dtype=np.float32)
        log_pi = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        rewards = np.zeros((self.n_workers, self.worker_steps, 1), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        timesteps = np.zeros((self.n_workers, self.worker_steps, 1), dtype=np.float32)

        with torch.no_grad():

            for t in range(self.worker_steps):
                obs[:, t] = self.obs
                timesteps[:, t] = self.timestep
                s_ = np.concatenate((obs[:, t], timesteps[:, t]), axis=1)
                s_ = obs_to_torch(s_, self.device)
                mean, sigma, v = model.forward(s_)
                action_dist = torch.distributions.Normal(mean, sigma)

                values[:, t] = v.cpu().numpy()
                a = action_dist.sample()

                actions[:, t] = a.cpu().numpy()
                log_pi[:, t] = action_dist.log_prob(a).cpu().numpy()

                for i in range(self.T):
                    for w, worker in enumerate(self.workers):
                        worker.child.send(("step", actions[w, t, :, i]))

                    for w, worker in enumerate(self.workers):
                        self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                for worker in self.workers:
                    worker.child.send(("timestep", None))
                for i, worker in enumerate(self.workers):
                    self.timestep[i] = worker.child.recv()

            s_ = np.concatenate((obs[:, t], timesteps[:, t]), axis=1)
            s_ = obs_to_torch(s_, self.device)
            mean, sigma, v = model.forward(s_)
            values[:, self.worker_steps] = v.cpu().numpy()

        adv = self.gae(values, done, rewards)
        samples = {'adv': adv, 'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'timesteps': timesteps,
                   'values': values[:, :-1], 'rewards': rewards}

        samples_flat = self.pre_process(samples)
        return samples_flat

    def gae(self, values, done, rewards):
        gae = 0
        adv = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        value_step = values[:, -1]

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            mask = np.array([mask, ] * 2 * self.T).transpose().reshape(done.shape[0], 2, self.T)
            rewards_ = np.array([rewards[:, t], ] * 2 * self.T).transpose().reshape(rewards.shape[0], 2, self.T)
            delta = rewards_ + self.gamma * value_step * mask - values[:, t]
            gae = delta + self.gamma * self.lambdas * gae * mask
            adv[:, t] = gae
            value_step = values[:, t]
        return adv

    def pre_process(self, samples):
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            samples_flat[k] = obs_to_torch(v, self.device)
        return samples_flat

    def meta_objective(self, model, samples):
        # PPO objective
        values = samples['values']
        adv = samples['adv']

        values = values[:, :-1]
        commuted_returns = adv + values
        adv_normalized = (adv - adv.mean(axis=0)) / (adv.std(axis=0) + 1e-10)

        obs = samples['obs']
        timesteps = samples['timesteps']
        s_ = torch.cat((obs, timesteps), dim=1)
        mean, sigma, value = model.forward(s_)

        actions = samples['actions']
        log_pi_old = samples['log_pi_old']
        pi = torch.distributions.Normal(mean, sigma)
        log_pi_new = pi.log_prob(actions)

        # commute current policy and value
        ratio = torch.exp(log_pi_new - log_pi_old)
        p1 = ratio * adv_normalized
        p2 = ratio.clamp(min=1.0 - self.clip, max=1.0 + self.clip) * adv_normalized
        policy_loss = -torch.mean(torch.min(p1, p2))

        # clipped value loss ppo2
        v1 = (value - commuted_returns) ** 2
        clipped = values + (value - values).clamp(min=-self.clip, max=self.clip)
        v2 = (clipped - commuted_returns) ** 2
        critic_loss = torch.mean(torch.max(v1, v2))

        loss = policy_loss + 0.25 * critic_loss - 0.02 * (pi.entropy().mean())
        return loss

    def regular_train(self, loss_fn, task_model, learned):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        reward_trace = []
        loss_trace = []

        optimizer = torch.optim.SGD(task_model.parameters(), lr=self.inner_lr)
        self.reset_env()

        for n_iter in range(self.n_inner_itr):

            samples = self.rollout(task_model)
            obs = samples['obs']
            timesteps = samples['timesteps']
            s_ = torch.cat((obs, timesteps), dim=1)

            samples = self.rollout(task_model)
            rewards = samples['rewards']
            reward_trace.append(torch.sum(rewards, dim=1).mean().item())

            for epoch in range(self.epochs):
                optimizer.zero_grad()

                mean, sigma, value = task_model.forward(s_)
                if not learned:
                    loss = loss_fn(task_model, samples)
                else:
                    loss = loss_fn(mean, sigma, value, samples)

                loss.backward()
                optimizer.step()
                loss_trace.append(loss.item())

        return loss_trace, reward_trace

    def train_ml3(self):
        # -------------------initialization------------------
        self.initalize_env()
        # -----------------------Policy----------------------
        task_models = CNNndpPolicy().to(self.device)
        task_opts = torch.optim.SGD(task_models.parameters(), lr=self.inner_lr)
        torch.save(task_models.state_dict(), "task_" + 'model.mdl')
        # -----------------------ML3-loss--------------------
        meta_loss = PPORewardsLearning(in_dim=16).to(self.device)
        meta_opts = torch.optim.Adam(task_models.parameters(), lr=self.outer_lr)

        for outer_i in range(self.n_outer_itr):
            task_models.load_state_dict(torch.load("task_" + 'model.mdl'))
            self.reset_env()
            with higher.innerloop_ctx(task_models, task_opts,
                                      copy_initial_weights=False) as (fmodel, diffopt):

                for i_inner in range(self.n_inner_itr):
                    samples = self.rollout(fmodel)
                    obs = samples['obs']
                    timesteps = samples['timesteps']
                    s_ = torch.cat((obs, timesteps), dim=1)

                    for epoch in range(self.epochs):
                        mean, sigma, value = fmodel.forward(s_)

                        pred_loss = meta_loss.forward(mean, sigma, value, samples)
                        diffopt.step(pred_loss)

                samples = self.rollout(fmodel)
                task_loss = self.meta_objective(fmodel, samples)
                task_loss.backward()
            meta_opts.step()

            if outer_i % 20 == 0:
                task_models.load_state_dict(torch.load("task_" + 'model.mdl'))
                loss_trace_learned, reward_learned = self.regular_train(meta_loss, task_models, True)
                task_models.load_state_dict(torch.load("task_" + 'model.mdl'))
                loss_trace_Reinf, reward_Reinf = self.regular_train(self.meta_objective, task_models, False)

                LOG.info(
                    f' [Epoch {outer_i:.2f}] ML3 Reward: {reward_learned[-1]:.2f}]| : PPO Reward {reward_Reinf[-1]:.2f}'
                )

                if self.plotting:
                    Y = np.stack([reward_learned, reward_Reinf]).transpose()
                    X = np.stack([range(0, len(reward_Reinf), 1), range(0, len(reward_Reinf), 1)]).transpose()

                    viz.line(Y=Y, X=X, env="rewards_ml3_reinf", win='eval_rewards',
                             opts=dict(showlegend=True, title='ml3/reinforce eval rewards',
                                       legend=['learned rewards', 'Standard rewards']))

                    viz.line([[reward_learned[-1], reward_Reinf[-1]]], [outer_i], env="rewards_ml3_reinf",
                             win='rewards_update_steps', update='append',
                             opts=dict(showlegend=True, title='Final rewards at n_outer_iters',
                                       legend=['learned rewards', 'Standard rewards']))

    def store_model(self, model, name):
        torch.save(model.state_dict(), name + '_model.mdl')

    def load_model(self, model, path):
        weights = torch.load(path, map_location=self.device)
        model.load_state_dict(weights, strict=False)
        return "model_loaded"


def main():
    # Initialize the trainer
    m = Agent(seed=0)

    global viz
    viz = Visdom()
    env_name = 'ml3_smnist_env'

    if viz.check_connection():
        plotting = True
    else:
        plotting = False


    global LOG
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    # Run and monitor the experiment
    m.train_ml3()



# ## Run it
if __name__ == "__main__":
    main()
