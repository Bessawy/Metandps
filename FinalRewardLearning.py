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


def fanin_init(tensor):
    '''
    :param tensor: torch.tensor
    used to initialize network parameters
    '''
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def init(module, weight_init, bias_init, gain=1):
    '''
      :param tensor: torch.tensor
      used to initialize network parameters
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class CNNndprl(nn.Module):
    def __init__(self, state_space=2, action_space=2,
                 N=3, T=5, l=1, tau=1,
                 state_index=np.arange(2), b_init_value=0.1,
                 rbf='gaussian', az=True,
                 only_g=False):
        '''
        Deep Neural Network for ndp
        :param N: No of basis functions (int)
        :param state_index: index of available states (np.array)
        :param hidden_activation: hidden layer activation function
        :param b_init_value: initial bias value
        '''

        super(CNNndprl, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.T, self.N, self.l = T, N, l
        self.hidden = 64
        dt = 1.0 / (T * self.l)
        self.output_size = N * len(state_index) + len(state_index)

        self.state_index = state_index
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, a_z=az)
        self.func = DMPIntegrator(rbf=rbf, only_g=only_g, az=az)
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)

        self.conv1 = nn.Conv2d(1, 10, 5, 1)
        self.conv2 = nn.Conv2d(10, 10, 5, 1)

        self.fc1 = torch.nn.Linear(state_space + 1, self.hidden)
        fanin_init(self.fc1.weight)
        self.fc1.bias.data.fill_(b_init_value)

        feature_size = 10 * 4 * 4 + 3

        # actor
        self.fc2_mean = torch.nn.Linear(self.hidden, self.output_size)
        fanin_init(self.fc2_mean.weight)
        self.fc2_mean.bias.data.fill_(b_init_value)
        self.sigma = torch.nn.Linear(self.hidden, action_space)

        # critic
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)
        self.fc2_value = init(self.fc2_value, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

    def forward(self, state):
        x = state.view(-1, 3)
        x = self.fc1(x)
        x = torch.tanh(x) * 1e-1

        y0 = state[:, :2]
        y0 = y0.reshape(-1, 1)[:, 0]
        dy0 = torch.zeros_like(y0) + 0.01

        # critic for T actions
        value = self.fc2_value(x).repeat(1, self.T).view(-1, self.T, 2)
        value = torch.transpose(value, 1, 2)
        # sigma for T actions
        sigma = torch.sigmoid(self.sigma(x))*1.0 + 0.001
        sigma = sigma.view(-1,2,1)

        # actions

        ndp_wg = self.fc2_mean(x)
        y, dy, ddy = self.func.forward(ndp_wg, self.DMPp, self.param_grad, None, y0, dy0)  # y = [200,301]
        y = y.view(state.shape[0], len(self.state_index), -1)
        y = y[:, :, ::self.l]
        a = y[:, :, 1:] - y[:, :, :-1]

        return a, sigma, value



class Agent:
    def __init__(self, seed=0):
        # -----------------Loops--------------------
        self.n_outer_itr = 500
        self.n_inner_itr = 5
        self.epochs = 16
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
        #self.meta_network = PPORewardsLearning().to(self.device)
        #self.meta_opts = torch.optim.Adam(self.meta_network.parameters(), lr=self.outer_lr)
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

        for worker in self.workers:
            worker.child.send(("task_goal", None))
        for i, worker in enumerate(self.workers):
            self.target_goal[i] = worker.child.recv()

    def initalize_env(self, desired_idx=2):
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        self.obs = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.timestep = np.zeros((self.n_workers, 1), dtype=np.float32)
        self.target_goal = np.zeros((self.n_workers, 28, 28), dtype=np.float32)

    def rollout_no_grad(self, model, learned=False):
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
            adv_grad = self.gae_grad(values, done, task_goals)
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

    def gae_grad(self, values, done, task_goal):
        task_goal = task_goal.reshape(task_goal.shape[0] * task_goal.shape[1], *task_goal.shape[2:])
        task_goal = torch.tensor(task_goal, dtype=torch.float32).to(self.device)

        # commute rewards
        rewards = self.meta_network(task_goal)
        rewards = rewards.view(task_goal.shape[0], task_goal.shape[1], *task_goal.shape[2:])

        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        gae = torch.zeros_like(done[:,0], dtype=torch.float32).to(self.device)
        adv = torch.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=torch.float32).to(self.device)

        value_step = values[:, -1]

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            mask = mask.repeat(1, 2, 5)
            rewards_ = rewards[:, t]
            delta = rewards_ + self.gamma * value_step * mask - values[:, t]
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
        self.policy = CNNndprl().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.initalize_env(desired_idx=2)

        for n_outer in range(self.n_outer_itr):

            samples = self.rollout_no_grad(self.policy, learned=False)

            for i in range(self.epochs):

                loss = self.meta_objective(self.policy, samples)
                # Set learning rate

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

            if n_outer % 50 == 0:
                print("Update Model: ", n_outer)
                #self.store_model(update)
                self.eval()

    def store_model(self, model, name):
        torch.save(model.state_dict(), name + '_model.mdl')

    def load_model(self, model, path):
        weights = torch.load(path, map_location=self.device)
        model.load_state_dict(weights, strict=False)
        return "model_loaded"

    def eval(self):
        env = SMNIST()
        s_0 = env.reset()
        t_0 = env.timestep
        task_goal = env.task_goal

        trajecx = []
        trajecy = []
        rewards_ = []

        with torch.no_grad():

            for i in range(10):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = np.concatenate((s_0, [t_0]))
                input = np.array([input])
                input = obs_to_torch(input, self.device)
                a, s, v = self.policy.forward(input)
                a = a.detach().cpu().numpy()
                for i in range(self.T):
                    trajecx.append(s_0[0])
                    trajecy.append(s_0[1])
                    s1, rewards, done, info = env.step(a[:, :, i])
                    s_0 = s1.squeeze()
                    t_0 = env.timestep
                    rewards_.append(rewards)

            print("evalution rewards: ", sum(rewards_))
            fig, ax = plt.subplots()
            im = ax.imshow(env._task_goal)
            ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
            plt.savefig('digit.png')
            plt.clf()
            plt.close(fig)

def main():
    # Initialize the trainer
    m = Agent(seed=0)

    global LOG
    logging.basicConfig(level=logging.INFO)
    LOG = logging.getLogger(__name__)

    # Run and monitor the experiment
    m.train_ml3()



# ## Run it
if __name__ == "__main__":
    main()
