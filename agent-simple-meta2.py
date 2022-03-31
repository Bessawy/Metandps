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
import higher

from smnistenv import Worker

def get_arguments():
    arg = {}
    arg['updates'] = 10000
    arg['epochs'] = 4  # number of epochs to train the model with sampled data
    arg['n_workers'] = 4 # number of worker processes
    arg['worker_steps'] = 10 # number of steps to run on each process for a single update
    arg['n_mini_batch'] = 4
    arg['lambdas'] = 0.96
    arg['gamma'] = 0.99
    arg['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arg['ppo_lr'] = 1e-3

    return arg

def obs_to_torch(obs: np.ndarray, device):
    return torch.FloatTensor(obs).to(device)

class MetaPolicy2(nn.Module):
    def __init__(self, state_space=2, action_space=2):
        super().__init__()
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space+4+2, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, 1)
        nn.init.uniform_(self.fc2.weight, a=0.0, b=1.0)

    def forward(self, traj, input, r, value_):
        input = input.view(-1, 2)
        traj = traj.view(-1,4)
        r = r.view(-1, 2)
        value_ = value_.view(-1,2)

        commute_return = torch.abs(r - value_)
        x = torch.cat((input, traj, commute_return), dim=1)

        x = self.fc1(x)
        x = F.relu(x) * 1e-2
        x = self.fc2(x)
        x = torch.abs(x)
        return x.mean()

class MetaPolicy(nn.Module):
    def __init__(self, state_space=2, action_space=2):
        super().__init__()
        self.hidden = 64
        self.
        nn.init.uniform_(self.fc2.weight, a=0.0, b=1.0)

    def forward(self, traj, input, r, value_):
        input = input.view(-1, 2)
        traj = traj.view(-1,4)
        r = r.view(-1, 2)
        value_ = value_.view(-1,2)

        commute_return = torch.abs(r - value_)
        x = torch.cat((input, traj, commute_return), dim=1)

        x = self.fc1(x)
        x = F.relu(x) * 1e-2
        x = self.fc2(x)
        x = torch.abs(x)
        return x.mean()



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
        
    def reset(self):
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
        mean, sigma = action_mean[:, :2], action_mean[:, 2:]
        sigma = torch.sigmoid(sigma) * 1.0 + 0.001


        value = self.fc2_value(x)
        return mean, sigma, value



def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)

class Agent:
    def __init__(self):

        self.updates = 10000
        self.epochs = 1
        self.n_workers = 4
        self.worker_steps = 150
        self.n_mini_batch = 1
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        self.rewards_history = []

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
        self.metaPolicy = MetaPolicy().to(self.device)
        self.policy = Policy().to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.metaPolicy.parameters(), lr=1e-3)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3, weight_decay=1e-5)
        self.episode = 0

        torch.save(self.policy.state_dict(), 'polivy.pth')


    def unroll(self, model):
        values = np.zeros((self.n_workers, self.worker_steps + 1, 2), dtype=np.float32)
        log_pi = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=bool)
        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        obs = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        goals = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)
        means = []
        sigmas = []
        value_ = []
        self.rewards_at = []



        for t in range(self.worker_steps):

            obs[:, t] = self.obs
            goals[:, t] = self.goal
            input = goals[:, t] - obs[:, t]

            mean, sigma, v = model.forward(obs_to_torch(input, self.device))
            means.append(mean)
            sigmas.append(sigma)
            value_.append(v)

            dists = torch.distributions.Normal(mean, sigma)

            values[:, t] = v.detach().cpu().numpy()
            a = dists.sample().detach()

            actions[:, t] = a.detach().cpu().numpy()
            log_pi[:, t] = dists.log_prob(a).detach().cpu().numpy()

            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()
                if done[w, t]:
                    self.episode += 1
                    self.rewards_history.append(info)
                    #if self.episode % 100 == 0:
                    #    print("Winrate for the last 100 episode: ", np.mean(self.rewards_history[-100:]))

            for worker in self.workers:
                worker.child.send(("goal", None))
            for i, worker in enumerate(self.workers):
                self.goal[i] = worker.child.recv()

        input = self.goal - self.obs
        a, s, v = model.forward(obs_to_torch(input, self.device))
        values[:, self.worker_steps] = v.detach().cpu().numpy()

        gae = 0
        adv = np.zeros((self.n_workers, self.worker_steps, 2), dtype=np.float32)

        value_step = values[:, -1]

        for t in reversed(range(self.worker_steps)):
            mask = 1.0 - done[:, t]
            mask = np.array([mask, ] * 2).transpose()
            rewards_ = np.array([rewards[:, t], ] * 2).transpose()
            delta = rewards_ + self.gamma * value_step * mask - values[:, t]
            gae = delta + self.gamma * self.lambdas * gae * mask
            adv[:, t] = gae
            value_step = values[:, t]

        samples = {'advantages': adv, 'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'goals': goals,
                   'values': values[:, :-1]}



        means = torch.stack(means, dim=1).view(-1, 2)
        sigmas = torch.stack(sigmas, dim=1).view(-1, 2)
        value_ = torch.stack(value_, dim=1).view(-1, 2)
        traj = torch.stack((means, sigmas), dim=1).view(-1,4)

        episode_reward = np.sum(rewards, axis=1)
        episode_reward = np.mean(episode_reward, axis = 0)

        return samples, traj, value_, episode_reward

    def pre_process(self, samples):
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs' or k == 'goals':
                samples_flat[k] = obs_to_torch(v, self.device)
            else:
                samples_flat[k] = torch.tensor(v, device=self.device)

        obs = samples_flat['obs']
        goals = samples_flat['goals']
        action = samples_flat['actions']
        adv = samples_flat['advantages']
        values = samples_flat['values']
        log_pi_old = samples_flat['log_pi_old']

        return obs, goals, action, adv, values, log_pi_old

    def regular_train(self, task_model, loss_fn):
        # Set up lists t
        reward = []
        loss_trace = []
        # Define optimizer
        optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-3)
        for ep in range(20):
            optimizer.zero_grad()
            samples, traj, value_, episode_reward = self.unroll(task_model)
            obs, goals, action, adv, values, log_pi = self.pre_process(samples)
            r = adv + values
            input = goals - obs
            loss = loss_fn(traj, input, r, value_)
            loss.backward()
            optimizer.step()

            loss_trace.append(loss.item())
            reward.append(episode_reward)


        self.eval(task_model)
        average = np.mean(reward)
        return average


    def meta_objective(self, model, states, actions, rewards):
        # propagate through the higher fmodel once again?
        means, sigs = model(states)
        # create distributions and extract logprob of the
        # used in query unroll actions ?
        dists = torch.distributions.Normal(means, sigs)
        logprob = dists.log_prob(actions)
        # now we extract the negative log loss
        reward_tensor = torch.FloatTensor(rewards)
        selected_logprobs = reward_tensor * logprob.sum(dim=-1)
        return -selected_logprobs.mean()

    def train(self):
        for update in range(self.updates):
            learnrate=1e-3
            clip=0.1
            step_reward = []
            self.meta_optimizer.zero_grad()
            self.policy.load_state_dict(torch.load('polivy.pth'))
            with higher.innerloop_ctx(self.policy, self.optimizer,
                        copy_initial_weights=False) as (fmodel, diffopt):

                samples, traj, value_, episode_reward = self.unroll(fmodel)
                obs, goals, action, adv, values, log_pi_oldm = self.pre_process(samples)

                for i in range(20):
                    r = adv + values
                    input = goals - obs
                    pred_loss = self.metaPolicy(traj, input, r, value_)
                    diffopt.step(pred_loss)

                    step_reward.append(episode_reward)

                    samples, traj, value_, episode_reward = self.unroll(fmodel)
                    obs, goals, action, adv, values, log_pi_old = self.pre_process(samples)

                commuted_returns = adv + values
                adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-10)
                input = goals - obs

                mean, std, value = fmodel.forward(input)
                dists = torch.distributions.Normal(mean, std)
                log_pi_new = dists.log_prob(action)

                ratio = torch.exp(log_pi_new - log_pi_old)
                p1 = ratio * adv_normalized
                p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                policy_loss = -torch.mean(torch.min(p1, p2))

                # clipped value loss ppo2
                v1 = (value - commuted_returns) ** 2
                clipped = values + (value - values).clamp(min=-clip, max=clip)
                v2 = (clipped - commuted_returns) ** 2
                critic_loss = torch.mean(torch.max(v1, v2))

                task_loss = policy_loss + 0.25 * critic_loss - 0.02 * (dists.entropy().mean())
                task_loss.backward()

                self.episode += 1
                self.rewards_history.append(np.mean(step_reward))
                if self.episode % 100 == 0:
                    print("Winrate for the last 100 episode: ", np.mean(self.rewards_history[-10:]))

            self.meta_optimizer.step()

            if update % 10 == 0:
                print("Update Model: ", update)
                self.store_model(update)
                #self.eval()
                avr_re = self.regular_train(self.policy, self.metaPolicy)
                print("avearage test reward: ", avr_re)





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

    def eval(self, model):
        env = SMNIST()
        s_0 = env.reset()
        g_0 = env.goal

        trajecx = []
        trajecy = []
        rewards_ = []

        with torch.no_grad():

            for i in range(150):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = (g_0 - s_0)
                mean, s, v = model.forward(obs_to_torch(input, self.device))
                a = mean.detach().cpu().numpy()
                s1, rewards, done, info = env.step(a)
                s_0 = s1.squeeze()
                g_0 = env.goal
                rewards_.append(rewards)

            print("evalution rewards: ", sum(rewards_))
            fig, ax = plt.subplots()
            im = ax.imshow(env._task_goal)
            ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
            plt.savefig('digit.png')
            plt.clf()

            self.save_rewards_plt()

    def save_rewards_plt(self):
        plt.plot(self.rewards_history)
        plt.legend(['rewards'])
        plt.title("rewards for PPO during training")
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
