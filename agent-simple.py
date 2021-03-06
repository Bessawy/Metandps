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

from smnistenv import Worker


class Policy(torch.nn.Module):
    def __init__(self, state_space = 2, action_space = 2):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)

        self.fc2_mean = torch.nn.Linear(self.hidden, 2*action_space)
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)

        self.var = torch.nn.Parameter(torch.tensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.view(-1,2)
        x = self.fc1(x)
        x = F.relu(x)*1e-2

        action_mean = self.fc2_mean(x)
        mean, sigma = action_mean[:, :2], torch.exp(action_mean[:, 2:])
        sigma = torch.clamp(sigma, 0.001, 2)

        value = self.fc2_value(x)
        return mean, sigma, value


def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)


class Agent:
    def __init__(self):
        self.update_reward = []
        self.update_steps = []

        self.updates = 16
        self.epochs = 32
        self.n_workers = 4
        self.worker_steps = 300
        self.inner_itr = 10

        self.n_mini_batch = 1
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        self.lambdas = 0.96
        self.gamma = 0.99
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        self.obs = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.goal = np.zeros((self.n_workers, 2), dtype=np.float32)

        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        for worker in self.workers:
            worker.child.send(("goal", None))
        for i, worker in enumerate(self.workers):
            self.goal[i] = worker.child.recv()

        self.device = self.device()
        self.policy = Policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

        self.episode = 0
        self.rewards_history = []

    def rollout(self, model):
        values = np.zeros((self.n_workers, self.worker_steps + 1, 2), dtype=np.float32)
        log_pi = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        goals = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)

        for t in range(self.worker_steps):
            obs[:, t] = self.obs
            goals[:, t] = self.goal

            Input_to_policy = obs_to_torch(goals[:, t] - obs[:, t], self.device)
            mean, sigma, v = model.forward(Input_to_policy)
            action_dist = torch.distributions.Normal(mean, sigma)

            values[:, t] = v.detach().cpu().numpy()

            a = action_dist.sample()
            actions[:, t] = a.detach().cpu().numpy()
            log_pi[:, t] = action_dist.log_prob(a).detach().cpu().numpy()

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()
                if done[w, t]:
                    self.episode += 1

            for worker in self.workers:
                worker.child.send(("goal", None))
            for i, worker in enumerate(self.workers):
                self.goal[i] = worker.child.recv()

        input_to_policy = obs_to_torch(self.goal - self.obs, self.device)
        m, s, v = model.forward(input_to_policy)
        values[:, self.worker_steps] = v.detach().cpu().numpy()

        adv = self.gae(done, rewards, values)

        samples = {'advantages': adv, 'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'goals': goals,
                   'values': values[:, :-1]}

        return self.pre_processing(samples)

    def pre_processing(self, samples):
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs' or k == 'goals':
                samples_flat[k] = obs_to_torch(v, self.device)
            else:
                samples_flat[k] = torch.tensor(v, device=self.device)

        return samples_flat

    def gae(self, done, rewards, values):
        value_step = values[:, -1]
        adv = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        gae = 0

        #print("gae")
        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            mask = np.array([mask, ] * 2).transpose()
            rewards_ = np.array([rewards[:, t], ] * 2).transpose()
            delta = rewards_ + self.gamma * value_step * mask - values[:, t]
            gae = delta + self.gamma * self.lambdas * gae * mask
            adv[:, t] = gae
            value_step = values[:, t]

        return adv

    def train(self):
        for update in range(self.updates):
            clip = 0.1

            samples_flat = self.rollout(self.policy)

            for i in range(self.epochs):
                # shuffle for each epoch
                indexes = torch.randperm(self.batch_size)
                # for each mini batch
                for start in range(0, self.batch_size, self.mini_batch_size):
                    # get mini batch
                    end = start + self.mini_batch_size
                    mini_batch_indexes = indexes[start: end]
                    mini_batch = {}
                    for k, v in samples_flat.items():
                        mini_batch[k] = v[mini_batch_indexes]

                    obs = mini_batch ['obs']
                    goals = mini_batch['goals']
                    action = mini_batch ['actions']
                    adv = mini_batch['advantages']
                    values = mini_batch['values']
                    log_pi_old = mini_batch ['log_pi_old']

                    commuted_returns = adv + values
                    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-10)
                    inputs = goals - obs
                    mean, sigma, value = self.policy.forward(inputs)

                    pi = torch.distributions.Normal(mean, sigma)
                    log_pi_new = pi.log_prob(action)

                    ratio = torch.exp(log_pi_new - log_pi_old)
                    p1 = ratio * adv_normalized
                    p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                    policy_loss = -torch.mean(torch.min(p1, p2))

                    v1 = torch.abs(value - commuted_returns)
                    clipped = values + (value - values).clamp(min=-clip, max=clip)
                    v2 = torch.abs(clipped - commuted_returns)
                    critic_loss = torch.mean(torch.max(v1, v2))

                    loss = policy_loss + critic_loss  - 0.2 * (pi.entropy().mean())

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

            if update % 5 == 0:
                print("Update Model: ", update)
                self.update_steps.append(update)
                self.eval()




    def eval(self):
        env = SMNIST()
        s_0 = env.reset()
        g_0 = env.goal

        trajecx = []
        trajecy = []
        rewards_ = []

        with torch.no_grad():

            for i in range(300):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = (g_0 - s_0)
                mean, sigma, v = self.policy.forward(obs_to_torch(input, self.device))
                a = mean
                a = a.detach().cpu().numpy()
                s1, rewards, done, info = env.step(a)
                s_0 = s1.squeeze()
                g_0 = env.goal
                rewards_.append(rewards)

            print("evalution rewards: ", sum(rewards_))
            self.update_reward.append(sum(rewards_))
            fig, ax = plt.subplots()
            im = ax.imshow(env._task_goal)
            ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
            plt.savefig('digit.png')
            plt.clf()

            #self.save_rewards_plt()

    def save_rewards_plt(self, plot1, plot2=None):
        plt.plot(self.update_steps, self.update_reward)
        plt.legend(['rewards'])
        plt.xlabel('updates')
        plt.ylabel('rewards')
        plt.title("rewards for PPO during training")
        plt.savefig('rewards.png')
        plt.clf()

    def store_model(self, name):
        torch.save(self.policy.state_dict(), name + '.pth')

    def load_model(self, path):
        weights = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(weights, strict=False)
        return "model_loaded"

    def destroy(self):
        for worker in self.workers:
            worker.child.send(("close", None))

    def device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print(device)
        return device


def main():
    m = Agent()
    m.train()
    m.destroy()

# ## Run it
if __name__ == "__main__":
    main()
