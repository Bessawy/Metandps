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


def obs_to_torch(obs: np.ndarray, device):
    return torch.FloatTensor(obs).to(device)

class MetaLoss(nn.Module):
    def __init__(self, state_space=2, action_space=2):
        super().__init__()
        self.phi = torch.nn.Parameter(torch.tensor([1.0]))
        self.phi2 = torch.nn.Parameter(torch.tensor([0.25]))
        self.phi3 = torch.nn.Parameter(torch.tensor([0.02]))

    def forward(self, policy_loss, critic_loss, dists):
        loss = (self.phi * policy_loss + self.phi2 * critic_loss - self.phi3 * (dists.entropy().mean()))
        return loss

class MetaLoss2(nn.Module):
    def __init__(self, state_space=2, action_space=2):
        super().__init__()
        self.hidden_activation = F.relu
        self.fc1 = nn.Linear(2*5, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, state, mean, std, adv, value, commuted_return, actions, log_pi_old):

        y = torch.cat((state, mean, std, adv, actions), dim=1)*1e-3
        y = torch.tanh(self.fc1(y))
        y = self.fc2(y)

        dists = torch.distributions.Normal(mean, std)
        log_pi_new = dists.log_prob(actions)
        ratio = torch.exp(log_pi_new - log_pi_old)
        new_adv = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-10)
        p1 = ratio * new_adv
        p2 = ratio.clamp(min=1.0 - 0.1, max=1.0 + -0.1) * new_adv
        policy_loss = -torch.mean(torch.min(p1, p2))

        return policy_loss + 0.25 * value - 0.02 * (dists.entropy().mean())


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
        self.n_workers = 6
        self.worker_steps = 300
        self.n_mini_batch = 1
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        self.rewards_history = []

        self.inner_itr = 10
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
        self.metaloss = MetaLoss2().to(self.device)
        self.policy = Policy().to(self.device)
        self.meta_optimizer = torch.optim.Adam(self.metaloss.parameters(), lr=1e-4)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
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
        reward = []
        loss_trace = []
        clip = 0.1

        optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-3)

        with higher.innerloop_ctx(task_model, optimizer,
                                  copy_initial_weights=False) as (fmodel, diffopt):
            for ep in range(self.inner_itr):
                optimizer.zero_grad()
                samples, traj, value_, episode_reward = self.unroll(fmodel)
                obs, goals, action, adv, values, log_pi_old = self.pre_process(samples)
                commuted_returns = adv + values
                inputs = goals - obs

                adv_normalized = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-10)
                inputs = goals - obs

                for i in range(self.epochs):
                    mean, std, value = fmodel.forward(inputs)
                    dists = torch.distributions.Normal(mean, std)
                    log_pi_new = dists.log_prob(action)

                    ratio = torch.exp(log_pi_new - log_pi_old)
                    p1 = ratio * adv_normalized
                    p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                    policy_loss = -torch.mean(torch.min(p1, p2))

                    v1 = (value - commuted_returns) ** 2
                    clipped = values + (value - values).clamp(min=-clip, max=clip)
                    v2 = (clipped - commuted_returns) ** 2
                    critic_loss = torch.mean(torch.max(v1, v2))

                    loss = loss_fn(inputs, mean, std, adv_normalized, critic_loss, commuted_returns, action, log_pi_old)
                    diffopt.step(loss)

            torch.save(fmodel.state_dict(), 'policy2.pth')

        task_model.load_state_dict(torch.load('policy2.pth'))
        rewards = self.eval(task_model, "custom")
        return rewards

    def ppo_train(self, task_model):
        reward = []
        loss_trace = []
        clip = 0.1

        optimizer = torch.optim.Adam(task_model.parameters(), lr=1e-3)


        for ep in range(self.inner_itr):
            samples, traj, value_, episode_reward = self.unroll(task_model)
            obs, goals, action, adv, values, log_pi_old = self.pre_process(samples)
            commuted_returns = adv + values

            adv_normalized = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-10)
            inputs = goals - obs

            for i in range(self.epochs):
                optimizer.zero_grad()
                mean, std, value = task_model.forward(inputs)
                dists = torch.distributions.Normal(mean, std)
                log_pi_new = dists.log_prob(action)

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



        loss_trace.append(loss.item())
        reward.append(episode_reward)

        rewards = self.eval(task_model, "ppo")
        average = np.mean(reward)
        return rewards



    def eval_cases(self, task_model, loss_fn):
        #torch.manual_seed(0)
        #np.random.seed(0)

        task_model.load_state_dict(torch.load('polivy.pth'))
        rw_1 = self.regular_train(task_model, loss_fn)
        task_model.load_state_dict(torch.load('polivy.pth'))
        rw_2 = self.ppo_train(task_model)
        #self.save_rewards_plt("both", rw_1, rw_2)


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
            clip = 0.1
            step_reward = []
            self.meta_optimizer.zero_grad()
            self.policy.load_state_dict(torch.load('polivy.pth'))
            with higher.innerloop_ctx(self.policy, self.optimizer,
                        copy_initial_weights=False) as (fmodel, diffopt):

                samples, traj, value_, episode_reward = self.unroll(fmodel)
                obs, goals, action, adv, values, log_pi_old = self.pre_process(samples)

                for i in range(self.inner_itr):
                    commuted_returns = adv + values
                    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-10)
                    inputs = goals - obs

                    for i in range(self.epochs):
                        mean, std, value = fmodel.forward(inputs)
                        dists = torch.distributions.Normal(mean, std)
                        log_pi_new = dists.log_prob(action)

                        ratio = torch.exp(log_pi_new - log_pi_old)
                        p1 = ratio * adv_normalized
                        p2 = ratio.clamp(min=1.0 - clip, max=1.0 + clip) * adv_normalized
                        policy_loss = -torch.mean(torch.min(p1, p2))

                        v1 = (value - commuted_returns) ** 2
                        clipped = values + (value - values).clamp(min=-clip, max=clip)
                        v2 = (clipped - commuted_returns) ** 2
                        critic_loss = torch.mean(torch.max(v1, v2))
                        pred_loss = self.metaloss(inputs, mean, std, adv_normalized, critic_loss, commuted_returns, action, log_pi_old)
                        diffopt.step(pred_loss)

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

                print("episode_reward: ", episode_reward, " task_loss: ", task_loss)
                task_loss.backward()

                #self.episode += 1
                #self.rewards_history.append(np.mean(step_reward))
                #if self.episode % 2== 0:
                 #   print("reward for the last 100 episode: ", np.mean(self.rewards_history[-2:]))

            self.meta_optimizer.step()

            if update % 10 == 0:
                print("Update Model: ", update)
                #self.store_model(update)
                #self.eval()
                avr_re = self.eval_cases(self.policy, self.metaloss)






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

    def eval(self, model, name):
        env = SMNIST()
        s_0 = env.reset()
        g_0 = env.goal

        trajecx = []
        trajecy = []
        reward_ = []
        rewards_ = []

        with torch.no_grad():

            for i in range(300):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = (g_0 - s_0)
                mean, s, v = model.forward(obs_to_torch(input, self.device))
                a = mean.detach().cpu().numpy()
                s1, rewards, done, info = env.step(a)
                s_0 = s1.squeeze()
                g_0 = env.goal
                reward_.append(rewards)
                rewards_.append(np.sum(reward_))

            print("evalution rewards " + name +": ", sum(reward_))
            fig, ax = plt.subplots()
            im = ax.imshow(env._task_goal)
            ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
            plt.savefig(name+'digit.png')
            plt.clf()

            self.save_rewards_plt_(name, rewards_)
        return rewards_


    def save_rewards_plt_(self, name, rewards_):
        plt.plot(rewards_)
        plt.legend(['rewards'])
        plt.title("rewards during training")
        plt.savefig(name+'rewards.png')
        plt.clf()

    def save_rewards_plt(self, name, rewards_, rewards_2):
        plt.plot(rewards_)
        plt.plot(rewards_2)
        plt.legend(["ml3", "pp0"], loc ="lower right")
        plt.title("rewards for PPO and Meta")
        plt.savefig(name+'rewards.png')
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
