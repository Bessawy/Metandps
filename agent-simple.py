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

def get_arguments():
    arg = {}
    arg['updates'] = 10000
    arg['epochs'] = 4  # number of epochs to train the model with sampled data
    arg['n_workers'] = 4 # number of worker processes
    arg['worker_steps'] = 128 # number of steps to run on each process for a single update
    arg['n_mini_batch'] = 4
    arg['lambdas'] = 0.96
    arg['gamma'] = 0.99
    arg['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arg['ppo_lr'] = 1e-3

    return arg

def obs_to_torch(obs: np.ndarray, device):
    return torch.FloatTensor(obs).to(device)


class Policy(torch.nn.Module):
    def __init__(self, state_space = 2, action_space = 2):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.hidden = 32
        self.fc1 = torch.nn.Linear(state_space, self.hidden)

        self.fc2_mean = torch.nn.Linear(self.hidden, action_space * 2)
        self.fc2_value = torch.nn.Linear(self.hidden, action_space)

        self.var = torch.nn.Parameter(torch.tensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # Common part
        x = self.fc1(x)
        x = F.relu(x)

        # Actor part
        action_mean = self.fc2_mean(x)
        mean, sigma = action_mean[:, :2], torch.exp(action_mean[:, 2:])
        #sigma = self.var

        # Critic part
        value = self.fc2_value(x)

        action_dist = torch.distributions.Normal(mean, sigma)
        return action_dist, value



def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device) /255.0

class Agent:
    def __init__(self):
        # number of updates
        self.updates = 10000
        # number of epochs to train the model with sampled data
        self.epochs = 4
        # number of worker processes
        self.n_workers = 4
        # number of steps to run on each process for a single update
        self.worker_steps = 128
        # number of mini batches
        self.n_mini_batch = 4
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        # win-history
        self.rewards_history = []

        #Gae parameters
        self.lambdas = 0.96
        self.gamma = 0.99
        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        # initialize workers for observations


        self.obs = np.zeros((self.n_workers, 2), dtype=np.uint8)
        self.goal = np.zeros((self.n_workers, 2), dtype=np.uint8)


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

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=2e-4)

        self.episode = 0

    def train(self):
        for update in range(self.updates):

            progress = update / self.updates
            learnrate = 1e-3
            #learnrate = 1e-3 * (1 - progress)
            clip = 0.1 * (1 - progress)
            clip = 0.1

            values = np.zeros((self.n_workers, self.worker_steps + 1, 2), dtype=np.float32)
            log_pi = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
            done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
            rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
            actions = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.int32)
            obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
            goals = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)

            with torch.no_grad():
                for t in range(self.worker_steps):
                    obs[:, t] = self.obs
                    goals[:, t] = self.goal

                    #input = goals[:, t] - obs[:, t]
                    input = obs[:, t]

                    # sample action
                    pi, v = self.policy.forward(obs_to_torch(input, self.device))
                    # pi2, _ = self.policy2.forward(obs_to_torch(self.obs2))

                    values[:, t] = v.cpu().numpy()

                    a = pi.sample()

                    actions[:, t] = a.cpu().numpy()
                    log_pi[:, t] = pi.log_prob(a).cpu().numpy()

                    # run sampled actions on each worker
                    for w, worker in enumerate(self.workers):
                        worker.child.send(("step", actions[w, t]))

                    # get results after executing the actions

                    for w, worker in enumerate(self.workers):
                        self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()
                        if done[w, t]:
                            self.episode += 1
                            self.rewards_history.append(info)
                            if self.episode % 500 == 0:
                                print("Winrate for the last 100 episode: ", np.mean(self.rewards_history[-100:]))

                    for worker in self.workers:
                        worker.child.send(("goal", None))
                    for i, worker in enumerate(self.workers):
                        self.goal[i] = worker.child.recv()

                # Get value of after the final step

                #input = self.goal - self.obs
                input = self.obs
                _, v = self.policy.forward(obs_to_torch(input, self.device))
                values[:, self.worker_steps] = v.cpu().numpy()

            # calculate advantages for all samples
            gae = 0
            adv = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)

            # value(t+1) for all workers
            value_step = values[:, -1]

            # we go in the reverse order with the number of worker step we have
            for t in reversed(range(self.worker_steps)):
            # mask determine the termination of episode, if done mask is equal zero and
            # thus next step is zero
                mask = 1.0 - done[:, t]


                mask = np.array([mask,]*2).transpose()
                rewards_ = np.array([rewards[:, t], ] * 2).transpose()

                delta = rewards_ + self.gamma * value_step * mask - values[:, t]
                # gae(t) from gae(t+1)
                gae = delta + self.gamma * self.lambdas * gae * mask
                # save for each time step
                adv[:, t] = gae
                value_step = values[:, t]

            samples = {'advantages': adv, 'actions': actions,'log_pi_old': log_pi,'obs': obs, 'goals':goals,
                       'values': values[:, :-1]}

            # samples are currently in `[workers, time_step]` table,
            # we should flatten it for training
            samples_flat = {}
            for k, v in samples.items():
                v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                if k == 'obs' or k=='goals':
                    samples_flat[k] = obs_to_torch(v, self.device)
                else:
                    samples_flat[k] = torch.tensor(v, device=self.device)


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

                    # commuted return
                    commuted_returns = adv + values

                    # normalize adv
                    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-10)
                    # commute current policy and value

                    #input = goals - obs
                    input = obs
                    pi, value = self.policy.forward(input)
                    # commute old log policy
                    log_pi_new = pi.log_prob(action)

                    ratio = torch.exp(log_pi_new - log_pi_old)
                    p1 = ratio * adv_normalized
                    p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                    policy_loss = -torch.mean(torch.min(p1, p2))

                    # clipped value loss ppo2
                    v1 = (value - commuted_returns) ** 2
                    clipped = values + (value - values).clamp(min=-clip, max=clip)
                    v2 = (clipped - commuted_returns) ** 2
                    critic_loss = torch.mean(torch.max(v1, v2))

                    loss = policy_loss + 0.25 * critic_loss - 0.02 * (pi.entropy().mean())

                    # Set learning rate
                    for mod in self.optimizer.param_groups:
                        mod['lr'] = learnrate

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if update % 500 == 0:
                print("Update Model: ", update)
                self.store_model(update)
                self.optimizer.step()


    def get_name(self):
        return "NotsoWise"

    def store_model(self, it):
        torch.save(self.policy.state_dict(), str(it) + 'modellast.mdl')

    def load_model(self):
        #tested and it loads model from cpu and gpu normally
        weights = torch.load("9500modellast.mdl", map_location=self.device)
        #weight2 = torch.load("13500modelfinal.mdl")
        self.policy.load_state_dict(weights, strict=False)
        #self.policy2.load_state_dict(weights, strict=False)
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

    def eval(self):
        env = SMNIST()
        s_0 = env.reset()
        g_0 = env.goal

        trajecx = []
        trajecy = []

        for i in range(301):
            trajecx.append(s_0[0])
            trajecy.append(s_0[1])
            input = (g_0 - s_0)
            pi, v = self.policy.forward(obs_to_torch(input, self.device))
            a = pi.sample()
            s1, rewards, done, info = env.step(a)
            s_0 = s1
            g_0 = env.goal



def main():
    # Initialize the trainer
    m = Agent()
    # Run and monitor the experiment
    m.train()
    # Stop the workers
    m.destroy()

def test_model():
    # Initialize the trainer
    m = Agent()
    m.load_model()


# ## Run it
if __name__ == "__main__":
    main()
