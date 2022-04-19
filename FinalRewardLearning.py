import matplotlib.pyplot as plt
import multiprocessing
import multiprocessing.connection
import numpy as np
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import sys
from smnistenv2 import Worker
from smnistenv2 import SMNIST
from utils import *
from Policies.optimizee import *
from Policies.Learned_loss import *
import logging
from visdom import Visdom
import higher



class Agent:
    def __init__(self, seed=0):
        # -----------------Loops--------------------
        self.n_outer_itr = 10000
        self.n_inner_itr = 10
        self.episode_length = 10
        self.epochs = 8
        # -----------------Workers------------------
        self.n_workers = 6
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
        # -----------------------ML3-loss--------------------
        self.meta_network = PPORewardsLearning().to(self.device)
        self.meta_opts = torch.optim.Adam(self.meta_network.parameters(), lr=self.outer_lr)
        # -----------------Testing------------------
        self.seed = seed
        self.plotting = True
        self.reward_learned = []
        self.reward_normal = []

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

        for worker in self.workers:
            worker.child.send(("task_goal", None))
        for i, worker in enumerate(self.workers):
            self.target_goal[i] = worker.child.recv()

    def initalize_env(self, desired_idx=2):
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        self.obs = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.timestep = np.zeros((self.n_workers, 1), dtype=np.float32)
        self.target_goal = np.zeros((self.n_workers, 28, 28), dtype=np.float32)

    def rollout_no_grad(self, model, learned=False, train=False):
        '''
        :param model: Policy used to rollout
        :return: dict (no_grad)
        '''
        self.reset_env()

        values = np.zeros((self.n_workers, self.worker_steps + 1, 2, self.T), dtype=np.float32)
        log_pi = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        rewards = np.zeros((self.n_workers, self.worker_steps, 1), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        timesteps = np.zeros((self.n_workers, self.worker_steps, 1), dtype=np.float32)
        task_goals = np.zeros((self.n_workers, self.worker_steps, 28, 28), dtype=np.float32)

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

                if learned:
                    # store images to commute learned rewards
                    for worker in self.workers:
                        worker.child.send(("task_goal", None))
                    for i, worker in enumerate(self.workers):
                        self.target_goal[i] = worker.child.recv()

                    task_goals[:, t] = self.target_goal

            s_ = np.concatenate((obs[:, t], timesteps[:, t]), axis=1)
            s_ = obs_to_torch(s_, self.device)
            mean, sigma, v = model.forward(s_)
            values[:, self.worker_steps] = v.cpu().numpy()

        if not learned:
            adv = self.gae_no_grad(values, done, rewards)
            samples = {'adv': adv, 'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'timesteps': timesteps,
                       'values': values[:, :-1]}
            samples_flat = pre_process(samples, self.device)

        else:
            adv_grad = self.gae_grad(values, done, task_goals, train=train)
            adv_grad_flat = adv_grad.view(adv_grad.shape[0] * adv_grad.shape[1], *adv_grad.shape[2:])
            samples = {'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'timesteps': timesteps,
                       'values': values[:, :-1]}

            samples_flat = pre_process(samples, self.device)
            samples_flat['adv'] = adv_grad_flat

        return samples_flat

    def gae_no_grad(self, values, done, rewards):
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

    def gae_grad(self, values, done, task_goal, train=False):
        task_goal = task_goal.reshape(task_goal.shape[0] * task_goal.shape[1], *task_goal.shape[2:])
        task_goal = torch.tensor(task_goal, dtype=torch.float32).to(self.device)

        # commute rewards
        rewards = self.meta_network(task_goal)

        if not train:
            rewards = rewards.detach()

        rewards = rewards.view(done.shape[0], done.shape[1], *rewards.shape[1:])
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        gae = torch.zeros((self.n_workers, 2, self.T), dtype=torch.float32).to(self.device)
        adv = torch.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=torch.float32).to(self.device)

        value_step = values[:, -1]

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            mask = mask.view(-1,1).repeat(2, 1, 5).transpose(0,1)
            rewards_ = rewards[:, t]
            delta = rewards_ + self.gamma * value_step * mask - values[:, t] #workers, 2, 5
            gae = delta + self.gamma * self.lambdas * gae * mask
            adv[:, t] = gae
            value_step = values[:, t]
        return adv

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

    def regular_train(self, model, name, digit=0, learned_reward=False):
        '''
        :param model: policy
        :param digit: desired_idx
        :param learned_reward: bool (True to used learned reward)
        :return: Eval_reward
        '''
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for n_inner in range(self.n_inner_itr):
            samples = self.rollout_no_grad(model, learned=learned_reward)
            for i in range(self.epochs):
                loss = self.meta_objective(model, samples)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        eval_reward = self.eval_policy(model, name)
        return eval_reward


    def eval_policy(self, model, name):
        env = SMNIST()
        s_0 = env.reset()
        t_0 = env.timestep
        task_goal = env.task_goal

        trajecx = []
        trajecy = []
        rewards_ = []

        with torch.no_grad():
            for i in range(self.episode_length):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = np.concatenate((s_0, [t_0]))
                input = np.array([input])
                input = obs_to_torch(input, self.device)
                a, s, v = model.forward(input)
                a = a.detach().cpu().numpy()
                for i in range(self.T):
                    trajecx.append(s_0[0])
                    trajecy.append(s_0[1])
                    s1, rewards, done, info = env.step(a[:, :, i])
                    s_0 = s1.squeeze()
                    t_0 = env.timestep
                    rewards_.append(rewards)

        show_trajectory(task_goal, trajecx, trajecy, name)
        return sum(rewards_)

    def ml3_train(self):
        self.initalize_env(desired_idx=0)

        policy = CNNndpPolicy().to(self.device)
        torch.save(policy.state_dict(), 'policy.mdl')
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

        for n_outer in range(self.n_outer_itr):

            policy.load_state_dict(torch.load('policy.mdl'))

            self.meta_opts.zero_grad()
            with higher.innerloop_ctx(policy, optimizer,
                                      copy_initial_weights=False) as (fmodel, diffopt):

                for n_inner in range(self.n_inner_itr):
                    # has reward gradient
                    samples = self.rollout_no_grad(fmodel, learned=True, train=True)
                    for ep in range(self.epochs):
                        loss = self.meta_objective(fmodel, samples)
                        diffopt.step(loss)

                samples = self.rollout_no_grad(fmodel, learned=False)
                for ep in range(self.epochs):
                    task_loss = self.meta_objective(fmodel, samples)
                    task_loss.backward(retain_graph=True)

                #if n_outer%81== 0 and n_outer!=0:
                #    torch.save(fmodel.state_dict(), 'policy.mdl')

            self.meta_opts.step()


            if n_outer%5 == 0:
                torch.save(self.meta_network.state_dict(), 'ml3_reward.mdl')
                policy.load_state_dict(torch.load('policy.mdl'))
                eval_reward_learned = self.regular_train(policy, "ml3", learned_reward=True)
                self.reward_learned.append(eval_reward_learned)
                policy.load_state_dict(torch.load('policy.mdl'))
                eval_reward_normal = self.regular_train(policy, 'norm',learned_reward=False)
                self.reward_normal.append(eval_reward_normal)

                LOG.info(
                    f' [Epoch {n_outer:.2f}] ML3 Reward: {eval_reward_learned:.2f}]| '
                    f': normal Reward {eval_reward_normal:.2f}'
                )

                Y = np.stack([self.reward_learned, self.reward_normal]).transpose()
                X = np.stack([range(0, len(self.reward_learned)*20, 20), range(0, len(self.reward_learned)*20, 20)]).transpose()

                viz.line(Y=Y, X=X, env="rewards_ml3", win='eval_rewards',
                         opts=dict(showlegend=True, title='learned vs design rewards',
                                   legend=['learned rewards', 'Standard rewards']))



def main():
    # Initialize the trainer
    m = Agent(seed=0)

    global viz
    viz = Visdom()
    env_name = 'reward_learning'

    if viz.check_connection():
        plotting = True
    else:
        plotting = False

    global LOG
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    m.ml3_train()




# ## Run it
if __name__ == "__main__":
    main()
