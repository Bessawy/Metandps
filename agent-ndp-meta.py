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

    def reset(self):
        self.init_weights()
        self.fc2_value = init(self.fc2_value, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        fanin_init(self.fc2_mean.weight)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


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

        return a, sigma, value


class MetaLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 11

        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 20, 5, 1)

        self.hidden = 200
        self.input = 3 + 9

        self.activation = torch.nn.Tanh()
        self.fc1 = torch.nn.Linear(self.input + 20 * 4 * 4, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3 = torch.nn.Linear(self.hidden, self.hidden)
        self.output = torch.nn.Linear(self.hidden, 1)


    def forward(self, x, image):
        image = image.view(-1, 1, 28, 28)
        x = x*1e-2

        y = F.relu(self.conv1(image))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.conv2(y))
        y = F.max_pool2d(y, 2, 2)

        y = y.view(-1, 4 * 4 * 20)
        x = torch.cat((x, y), dim=1)

        x = self.activation(self.fc1(x)*1e-2)
        x = self.activation(self.fc2(x)*1e-2)
        x = self.activation(self.fc3(x) * 1e-2)
        x = self.output(x)

        return x.mean()




def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)


class Agent:
    def __init__(self):
        self.task_digits = [0,1,2]
        self.inner_itr = 30
        self.updates = 400
        self.epochs = 4
        self.n_workers = 4
        self.worker_steps = 60
        self.n_mini_batch = 1
        self.T = 5
        self.rewards_history = []
        self.lambdas = 0.96
        self.gamma = 0.99
        self.clip = 0.1
        self.task_losses = []
        self.all_task = 0

        self.obs = np.zeros((self.n_workers, 2), dtype=np.float32)
        self.timestep = np.zeros((self.n_workers, 1), dtype=np.float32)
        self.target_goal = np.zeros((self.n_workers, 28, 28))
        self.goal = np.zeros((self.n_workers, 2), dtype=np.float32)

        self.device = self.device()
        self.episode = 0

        self.metaloss = MetaLoss().to(self.device)
        self.metaoptimizer = torch.optim.Adam(self.metaloss.parameters(), lr=1e-4)

    def close_env(self):
        for worker in self.workers:
            worker.child.send(("close", None))

    def reset_env(self, digit):
        self.workers = [Worker(47 + i, digit) for i in range(self.n_workers)]

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

        for worker in self.workers:
            worker.child.send(("goal", None))
        for i, worker in enumerate(self.workers):
            self.goal[i] = worker.child.recv()

    def gae_ndp(self, values, done, rewards):
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
            if k == 'obs' or k == 'timesteps' or k == 'task_goals':
                samples_flat[k] = obs_to_torch(v, self.device)
            else:
                samples_flat[k] = torch.tensor(v, device=self.device)
        return samples_flat

    def rollout(self, model):
        values = np.zeros((self.n_workers, self.worker_steps + 1, 2, self.T), dtype=np.float32)
        log_pi = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps, 2, self.T), dtype=np.float32)
        obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        timesteps = np.zeros((self.n_workers, self.worker_steps, 1), dtype=np.float32)
        task_goals = np.zeros((self.n_workers, self.worker_steps, 28, 28), dtype=np.float32)
        goals = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)


        with torch.no_grad():
            for t in range(self.worker_steps):
                obs[:, t] = self.obs
                timesteps[:, t] = self.timestep
                task_goals[:, t] = self.target_goal
                goals[:, t] = self.goal

                input = np.concatenate((obs[:, t], timesteps[:, t]), axis=1)
                input = obs_to_torch(input, self.device)
                image = obs_to_torch(task_goals[:, t], self.device)
                mean, sigma, v = model.forward(input, image)
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
                        if done[w, t]:
                            self.episode += 1
                            #self.rewards_history.append(info)
                            #if self.episode % 100 == 0:
                                #print("Winrate for the last 100 episode: ", np.mean(self.rewards_history[-20:]))

                for worker in self.workers:
                    worker.child.send(("timestep", None))
                for i, worker in enumerate(self.workers):
                    self.timestep[i] = worker.child.recv()

                for worker in self.workers:
                    worker.child.send(("goal", None))
                for i, worker in enumerate(self.workers):
                    self.goal[i] = worker.child.recv()

            input = np.concatenate((self.obs, self.timestep), axis=1)
            input = obs_to_torch(input, self.device)
            image = obs_to_torch(self.target_goal, self.device)
            m, s, v = model.forward(input, image)
            values[:, self.worker_steps] = v.cpu().numpy()

        adv = self.gae_ndp(values, done, rewards)
        samples = {'adv': adv, 'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'timesteps': timesteps,
                   'values': values[:, :-1], 'task_goals': task_goals, 'goals':goals}

        samples_flat = self.pre_process(samples)

        obs, task_goals, actions = samples_flat['obs'], samples_flat['task_goals'], samples_flat['actions']
        values, log_pi_old = samples_flat['values'], samples_flat['log_pi_old']
        adv, timesteps = samples_flat['adv'], samples_flat['timesteps']
        goals = samples_flat['goals']

        adv_normalized = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-10)
        return obs, task_goals, timesteps, actions, adv, adv_normalized, values, log_pi_old, goals


    def train(self):
        task_models = []
        task_opts = []

        for i in range(len(self.task_digits)):
            task_models.append(CNNndprl().to(self.device))  # Optimzee
            task_opts.append(torch.optim.Adam(task_models[i].parameters(), lr=1e-3))
            #torch.save(task_models[i].state_dict(), "task_" + str(i) + '_policy.mdl')


        for update in range(self.updates):
            print("update: ", update)


            for t in range(len(self.task_digits)):
                self.reset_env(self.task_digits[t])
                #task_models[t].load_state_dict(torch.load("task_" + str(t) + '_policy.mdl'))
                task_models[t].reset()

                with higher.innerloop_ctx(task_models[t], task_opts[t],
                                          copy_initial_weights=False) as (fmodel, diffopt):
                    obs, task_goals, timesteps, actions, adv, adv_normalized, values, log_pi_old, goals = self.rollout(
                        task_models[t])

                    for i in range(self.inner_itr):
                        for itr in range(self.epochs):
                            input = torch.cat((obs, timesteps), dim=1)
                            mean, sigma, value = fmodel.forward(input, task_goals)
                            commuted_returns = adv + values

                            com = torch.norm(value - commuted_returns, dim=2)
                            com = torch.norm(com, dim=1).view(-1,1)
                            sigma_ = torch.norm(sigma, dim=2).view(-1,2)
                            actions_ = torch.sum(actions, dim=2).view(-1,2)
                            mean_ = torch.sum(mean, dim=2).view(-1,2)
                            adv_ = torch.norm(adv, dim=2).view(-1,2)

                            x = torch.cat((mean_, sigma_, com, input, actions_, adv_), dim=1)
                            pred_loss = self.metaloss(x, task_goals)
                            diffopt.step(pred_loss)

                        obs, task_goals, timesteps, actions, adv, adv_normalized, values, log_pi_old, goals = self.rollout(
                            fmodel)

                    input = torch.cat((obs, timesteps), dim=1)
                    mean, sigma, value = fmodel.forward(input, task_goals)

                    mean = torch.sum(mean, dim=2).view(-1,2)

                    task_loss = torch.norm(mean + obs - goals)
                    print(task_loss)
                    self.task_losses.append(task_loss.item())

                    self.all_task += task_loss.item()
                    self.task_losses.append(task_loss.item())
                    task_loss.backward()

                self.close_env()
                #torch.save(task_models[t].state_dict(), "task_" + str(t) + '_policy.mdl')

            print("Task losses: ", self.all_task/len(self.task_digits))
            self.all_task = 0
            self.metaoptimizer.step()


            if update % 10 == 0:
                print("Update Model: ", update)
                self.store_model()
                self.save_losses_plt()
                self.test(5, self.metaloss)


    def test(self, digit, loss_fn):
        self.reset_env(digit)
        task_model = CNNndprl().to(self.device)
        rw_ml3 = self.regular_train(task_model, loss_fn, digit)
        task_model2 = CNNndprl().to(self.device)
        rw_ppo = self.ppo_train(task_model2, digit)
        self.close_env()
        self.save_rewards_plt("both", rw_ml3, rw_ppo)


    def regular_train(self, task_model, loss_fn, digit):
        reward, loss_trace = [], []
        optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-3)
        for ep in range(self.inner_itr):
            obs, task_goals, timesteps, actions, adv, adv_normalized, values, log_pi_old, goals = self.rollout(
                task_model)
            for i in range(self.epochs):
                optimizer.zero_grad()

                input = torch.cat((obs, timesteps), dim=1)
                mean, sigma, value = task_model.forward(input, task_goals)
                commuted_returns = adv + values

                com = torch.norm(value - commuted_returns, dim=2)
                com = torch.norm(com, dim=1).view(-1, 1)
                sigma_ = torch.norm(sigma, dim=2).view(-1, 2)
                actions_ = torch.sum(actions, dim=2).view(-1, 2)
                mean_ = torch.sum(mean, dim=2).view(-1, 2)
                adv_ = torch.norm(adv, dim=2).view(-1, 2)

                x = torch.cat((mean_, sigma_, com, input, actions_, adv_), dim=1)
                loss = loss_fn(x, task_goals)
                loss.backward()
                optimizer.step()

                mean_ = torch.sum(mean, dim=2).view(-1, 2)
                task_loss = torch.mean((mean_ + obs - goals) ** 2)
                loss_trace.append(task_loss.item())

        #print(loss_trace[-1])
        rewards = 0
        rewards = self.eval(task_model, digit, "ml3")
        return rewards

    def ppo_train(self, task_model, digit):
        reward = []
        loss_trace = []
        clip = 0.1

        optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-3)
        for ep in range(self.inner_itr):
            obs, task_goals, timesteps, actions, adv, adv_normalized, values, log_pi_old, goals = self.rollout(
                task_model)
            commuted_returns = adv + values

            adv_normalized = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-10)
            for i in range(self.epochs):
                optimizer.zero_grad()
                input = torch.cat((obs, timesteps), dim=1)
                mean, sigma, value = task_model.forward(input, task_goals)
                dists = torch.distributions.Normal(mean, sigma)
                log_pi_new = dists.log_prob(actions)

                ratio = torch.exp(log_pi_new - log_pi_old)
                p1 = ratio * adv_normalized
                p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                policy_loss = -torch.mean(torch.min(p1, p2))

                v1 = (value - commuted_returns) ** 2
                clipped = values + (value - values).clamp(min=-clip, max=clip)
                v2 = (clipped - commuted_returns) ** 2
                critic_loss = torch.mean(torch.max(v1, v2))

                loss = policy_loss + 0.25 * critic_loss - 0.02 * (dists.entropy().mean())
                loss.backward()
                optimizer.step()


        rewards = self.eval(task_model, digit, "ppo")
        return rewards


    def store_model(self):
        torch.save(self.metaloss.state_dict(), 'ml3_model.mdl')

    def load_model(self):
        weights = torch.load("9500modellast.mdl", map_location=self.device)
        self.policy.load_state_dict(weights, strict=False)
        return "model_loaded"

    def device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        print(device)
        return device

    def eval(self, model, digit, name):
        env = SMNIST(digit_idx=digit)
        s_0 = env.reset()
        t_0 = env.timestep
        task_goal = env.task_goal

        trajecx = []
        trajecy = []
        rewards_ = []

        with torch.no_grad():

            for i in range( self.worker_steps):
                input = np.concatenate((s_0, [t_0]))
                input = np.array([input])
                input = obs_to_torch(input, self.device)
                image = obs_to_torch(task_goal, self.device)
                mean, std, v = model.forward(input, image)
                a = mean
                a = a.detach().cpu().numpy()
                for i in range(self.T):
                    trajecx.append(s_0[0])
                    trajecy.append(s_0[1])
                    s1, rewards, done, info = env.step(a[:, :, i])
                    s_0 = s1.squeeze()
                    t_0 = env.timestep
                rewards_.append(rewards)

            print("evalution rewards " + name + " : ", sum(rewards_))
            fig, ax = plt.subplots()
            im = ax.imshow(env._task_goal)
            ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
            plt.savefig(name+'_digit.png')
            plt.clf()
            plt.close(fig)

        return rewards_


    def save_rewards_plt(self, name, rewards_, rewards_2):
        plt.plot(rewards_)
        plt.plot(rewards_2)
        plt.legend(["ml3", "pp0"], loc="lower right")
        plt.ylim([0,2])
        plt.title("rewards for PPO and Meta")
        plt.savefig(name + 'rewards.png')
        plt.clf()

    def save_losses_plt(self,):
        plt.plot(self.task_losses)
        plt.legend(['loss'])
        plt.xlabel('ml3 update steps')
        plt.ylabel('losses')
        plt.ylim([0, 500])
        plt.title("losses during ml3 training")
        plt.savefig('losses_eval.png')
        plt.clf()



def main():
    # Initialize the trainer
    m = Agent()
    m.train()
    m.destroy()

def test_model():
    # Initialize the trainer
    m = Agent()
    m.load_model()


# ## Run it
if __name__ == "__main__":
    main()
