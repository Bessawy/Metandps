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


def obs_to_torch(obs: np.ndarray, device):
    return torch.FloatTensor(obs).to(device)


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

        self.fc1 = torch.nn.Linear(state_space + 1 + 10 * 4 * 4, self.hidden)
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

    def forward(self, state, image):

        image = image.view(-1, 1, 28, 28)
        x = state.view(-1, 3)

        y = F.relu(self.conv1(image))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2, 2)

        y = y.view(-1, 4 * 4 * 10)
        x = torch.cat((x, y), dim=1)

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

        actions_dist = torch.distributions.Normal(a, sigma)

        return actions_dist, value



def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)

class Agent:
    def __init__(self):

        torch.autograd.set_detect_anomaly(True)
        # number of updates
        self.updates = 10000
        # number of epochs to train the model with sampled data
        self.epochs = 6
        # number of worker processes
        self.n_workers = 6
        # number of steps to run on each process for a single update
        self.worker_steps = 60
        # number of mini batches
        self.n_mini_batch = 4
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        # action_steps
        self.T = 5

        # win-history
        self.rewards_history = []

        #Gae parameters
        self.lambdas = 0.96
        self.gamma = 0.99
        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]
        # initialize workers for observations

        self.obs = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.timestep = np.zeros((self.n_workers, 1), dtype=np.float32)
        self.target_goal = np.zeros((self.n_workers, 28, 28))

        for worker in self.workers:
            worker.child.send(("task_goal", None))
        for i, worker in enumerate(self.workers):
            self.target_goal[i] = worker.child.recv()

        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        for worker in self.workers:
            worker.child.send(("timestep", None))
        for i, worker in enumerate(self.workers):
            self.timestep[i] = worker.child.recv()

        self.device = self.device()
        self.policy = CNNndprl().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        self.episode = 0

    def train(self):
        for update in range(self.updates):

            progress = update / self.updates
            #learnrate = 2e-4 * (1 - progress)
            #clip = 0.1 * (1 - progress)
            learnrate = 1e-3
            clip = 0.1

            values = np.zeros((self.n_workers, self.worker_steps + 1, 2, self.T), dtype=np.float32)
            log_pi = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
            done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
            rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
            actions = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
            obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
            timesteps = np.zeros((self.n_workers, self.worker_steps, 1), dtype=np.float32)
            task_goals = np.zeros((self.n_workers, self.worker_steps, 28, 28), dtype=np.float32)

            with torch.no_grad():
                for t in range(self.worker_steps):
                    obs[:, t] = self.obs
                    timesteps[:, t] = self.timestep
                    task_goals[:, t] = self.target_goal

                    input = np.concatenate((obs[:, t], timesteps[:, t]), axis = 1)
                    input = obs_to_torch(input, self.device)
                    image = obs_to_torch(task_goals[:, t], self.device)
                    # sample action
                    pi, v = self.policy.forward(input, image)
                    values[:, t] = v.cpu().numpy()


                    a = pi.sample()

                    actions[:, t] = a.cpu().numpy()
                    log_pi[:, t] = pi.log_prob(a).cpu().numpy()

                    for i in range(self.T):

                        # run sampled actions on each worker
                        for w, worker in enumerate(self.workers):
                            worker.child.send(("step", actions[w, t, :, i]))


                        # get results after executing the actions
                        for w, worker in enumerate(self.workers):
                            self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()
                            if done[w, t]:
                                self.episode += 1
                                self.rewards_history.append(info)
                                if self.episode % 1000 == 0:
                                    print("Winrate for the last 100 episode: ", np.mean(self.rewards_history[-1000:]))

                    for worker in self.workers:
                        worker.child.send(("timestep", None))
                    for i, worker in enumerate(self.workers):
                        self.timestep[i] = worker.child.recv()

                # Get value of after the final step

                input = np.concatenate((self.obs, self.timestep), axis=1)
                input = obs_to_torch(input, self.device)
                image = obs_to_torch(self.target_goal, self.device)
                _, v = self.policy.forward(input, image)
                values[:, self.worker_steps] = v.cpu().numpy()

            # calculate advantages for all samples
            gae = 0
            adv = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
            # value(t+1) for all workers
            value_step = values[:, -1]

            # we go in the reverse order with the number of worker step we have
            for t in reversed(range(self.worker_steps)):
            # mask determine the termination of episode, if done mask is equal zero and
            # thus next step is zero
                mask = 1.0 - done[:, t]

                mask = np.array([mask,]*2*self.T).transpose().reshape(done.shape[0], 2, self.T)
                rewards_ = np.array([rewards[:, t], ] * 2*self.T).transpose().reshape(rewards.shape[0], 2, self.T)

                delta = rewards_ + self.gamma * value_step * mask - values[:, t]
                # gae(t) from gae(t+1)
                gae = delta + self.gamma * self.lambdas * gae * mask
                # save for each time step
                adv[:, t] = gae
                value_step = values[:, t]

            samples = {'advantages': adv, 'actions': actions, 'log_pi_old': log_pi,'obs': obs, 'timesteps': timesteps,
                       'values': values[:, :-1], 'task_goals': task_goals}

            # samples are currently in `[workers, time_step]` table,
            # we should flatten it for training
            samples_flat = {}
            for k, v in samples.items():
                v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
                if k == 'obs' or k=='timesteps' or k=='task_goals':
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
                    timesteps = mini_batch['timesteps']
                    action = mini_batch ['actions']
                    adv = mini_batch['advantages']
                    values = mini_batch['values']
                    log_pi_old = mini_batch ['log_pi_old']
                    task_goals = mini_batch['task_goals']

                    # commuted return
                    commuted_returns = adv + values

                    # normalize adv
                    adv_normalized = (adv - adv.mean(axis = 0)) / (adv.std(axis = 0) + 1e-10)
                    # commute current policy and value

                    input = torch.cat((obs, timesteps), dim=1)
                    pi, value = self.policy.forward(input, task_goals)
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
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                    self.optimizer.step()

            if update % 1000 == 0:
                print("Update Model: ", update)
                self.store_model(update)
                self.eval()




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
        t_0 = env.timestep
        task_goal = env.task_goal

        trajecx = []
        trajecy = []
        rewards_ = []

        with torch.no_grad():

            for i in range(60):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = np.concatenate((s_0, [t_0]))
                input = np.array([input])
                input = obs_to_torch(input, self.device)
                image = obs_to_torch(task_goal, self.device)
                pi, v = self.policy.forward(input, image)
                a = pi.mean
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

            self.save_rewards_plt()

    def save_rewards_plt(self):
        plt.plot(self.rewards_history)
        plt.legend(['rewards'])
        plt.title("rewards for PPO during training")
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.savefig('rewards.png')
        plt.clf()



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
