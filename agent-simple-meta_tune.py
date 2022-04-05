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
import optuna
from smnistenv import Worker


class Policy(torch.nn.Module):
    def __init__(self, state_space = 2, action_space = 2):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)

        self.min_action, self.max_action = -5, 5

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
        mean, sigma = action_mean[:, :2], action_mean[:, 2:]
        sigma = torch.sigmoid(sigma)*2.0 + 1e-4
        value = self.fc2_value(x)
        mean = torch.tanh(mean)*4
        return mean, sigma, value


def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)

class MetaLoss(torch.nn.Module):
    def __init__(self, trial):
        super().__init__()

        activation_candidates = {
            "celu": torch.nn.ELU(),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
        }

        nonlinearity_name = trial.suggest_categorical("nonlinearity", list(activation_candidates))
        self.activation = activation_candidates[nonlinearity_name]

        n_layers = trial.suggest_int("n_layers", 2, 4)
        self.layers = torch.nn.ModuleList()

        print(n_layers, nonlinearity_name)

        in_dim = 11
        for i in range(n_layers):
            out_features = trial.suggest_int("n_units_l{}".format(i), 64, 256)
            self.layers.append(nn.Linear(in_dim, out_features))
            in_dim = out_features

        self.fc1 = nn.Linear(in_dim, 1)

        #self.loss_fn = torch.nn.Sequential(torch.nn.Linear(11, num_neurons),
        #                                   activation(),
        #                                   torch.nn.Linear(num_neurons, num_neurons),
        #                                   activation(),
        #                                   torch.nn.Linear(num_neurons, 1))


    def forward(self, x):
        for layer in self.layers:
             x = layer(x)
             x = self.activation(x)*1e-2

        x = self.fc1(x)
        return x.mean()

class Agent:
    def __init__(self, params, trial):
        self.update_reward = []
        self.update_steps = []
        self.task_losses = []

        self.updates = 500
        self.epochs = 32
        self.n_workers = 4
        self.worker_steps = 300
        self.inner_itr = 10
        self.N = 300

        self.n_mini_batch = 1
        self.batch_size = self.n_workers * self.worker_steps
        self.mini_batch_size = self.batch_size // self.n_mini_batch

        self.clip = 0.1
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
        self.metaloss = MetaLoss(trial).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
        #self.metaoptimizer = torch.optim.Adam(self.metaloss.parameters(), lr=1e-3)
        self.metaoptimizer = getattr(torch.optim, params['optimizer'])(self.metaloss.parameters(), lr=params['learning_rate'])

        torch.save(self.policy.state_dict(), 'polivy.pth')

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
        means, sigmas, value_ = [], [], []


        for t in range(self.worker_steps):
            obs[:, t] = self.obs
            goals[:, t] = self.goal

            Input_to_policy = obs_to_torch(goals[:, t] - obs[:, t], self.device)
            mean, sigma, v = model.forward(Input_to_policy)
            action_dist = torch.distributions.Normal(mean, sigma)

            means.append(mean)
            sigmas.append(sigma)
            value_.append(v)

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

        means = torch.stack(means, dim=1).view(-1, 2)
        sigmas = torch.stack(sigmas, dim=1).view(-1, 2)
        value_torch = torch.stack(value_, dim=1).view(-1, 2)

        adv = self.gae(done, rewards, values)
        samples = {'actions': actions, 'log_pi_old': log_pi, 'obs': obs, 'goals': goals,
                   'values': values[:, :-1], 'rewards': rewards, 'adv': adv}

        samples_flat = self.pre_processing(samples)

        obs, goals, actions = samples_flat['obs'], samples_flat['goals'], samples_flat['actions']
        values, log_pi_old = samples_flat['values'], samples_flat['log_pi_old']
        adv =  samples_flat['adv']

        adv_normalized = (adv - adv.mean(dim=0)) / (adv.std(dim=0) + 1e-10)
        end_reward = rewards.sum(axis=1).mean(axis=0)
        return obs, goals, actions, adv, adv_normalized, values, log_pi_old, means, sigmas, value_torch, end_reward

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

        return adv

    def train(self, trial):
        for update in range(self.updates):
            self.metaoptimizer.zero_grad()
            self.policy.load_state_dict(torch.load('policy2.pth'))
            #self.policy.reset()
            with higher.innerloop_ctx(self.policy, self.optimizer,
                    copy_initial_weights=False) as (fmodel, diffopt):

                obs, goals, actions, adv, adv_noparam, values, log_pi_old, means, sigmas, value_torch, end_r = self.rollout(fmodel)
                for k in range(self.inner_itr):
                    for i in range(self.epochs):
                        inputs = goals - obs
                        mean, std, value = fmodel.forward(inputs)
                        commuted_returns = adv + values
                        com = torch.norm(value - commuted_returns, dim=1).view(-1,1)
                        x = torch.cat((mean, std, com, inputs, actions, adv), dim=1)
                        pred_loss = self.metaloss(x)
                        diffopt.step(pred_loss)
                    obs, goals, actions, adv, adv_noparam, values, log_pi_old, means, sigmas, value_torch, end_r= self.rollout(fmodel)

                inputs = goals - obs
                mean, std, value = fmodel.forward(inputs)
                task_loss = torch.norm(mean + obs - goals)
                self.task_losses.append(task_loss.item())
                task_loss.backward()
            self.metaoptimizer.step()

            trial.report(self.task_losses[-1], update)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if update % 100 == 0:
            print(np.mean(self.task_losses[-20]))
            #    print("Update Model: ", update)
                #print("param: ", self.metaloss.param)
                #self.update_steps.append(update)
        #reward_to_maximize = self.test(self.policy, self.metaloss)
        return np.mean(self.task_losses[-10])


    def test(self, task_model, loss_fn):
        #task_model.reset()
        rw_ml3 = self.regular_train(task_model, loss_fn, "ml3")
        return rw_ml3

    def regular_train(self, task_model, loss_fn, name):
        reward, loss_trace = [], []
        for ep in range(self.inner_itr):
            obs, goals, actions, adv, adv2, values, log_pi_old, means, sigmas, value_torch, end_r = self.rollout(task_model)
            for i in range(self.epochs):
                self.optimizer.zero_grad()
                inputs = goals - obs
                mean, std, value = task_model.forward(inputs)
                commuted_returns = adv + values
                com = torch.norm(value - commuted_returns, dim=1).view(-1,1)
                x = torch.cat((mean, std, com, inputs, actions, adv), dim=1)
                loss = loss_fn(x)
                loss.backward()
                self.optimizer.step()

        rewards = self.eval_policy(task_model, name)
        return rewards


    def eval_policy(self, model, name):
        env = SMNIST()
        s_0, g_0 = env.reset(), env.goal
        trajecx, trajecy, rewards_ = [], [], []

        with torch.no_grad():
            for i in range(self.N):
                trajecx.append(s_0[0])
                trajecy.append(s_0[1])
                input = (g_0 - s_0)
                mean, sigma, v = self.policy.forward(obs_to_torch(input, self.device))
                a = mean.detach().cpu().numpy()
                s1, rewards, done, info = env.step(a)
                s_0, g_0 = s1.squeeze(), env.goal
                rewards_.append(rewards)


            print("evalution rewards: ", sum(rewards_))
            self.update_reward.append(sum(rewards_))

            fig, ax = plt.subplots()
            im = ax.imshow(env._task_goal)
            ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
            plt.savefig('digit.png')
            plt.clf()

            #self.save_rewards_plt()
            #self.save_losses_plt()

        return self.update_reward[-1]

    def save_rewards_plt(self,):
        plt.plot(self.update_steps, self.update_reward)
        plt.legend(['rewards'])
        plt.xlabel('updates')
        plt.ylabel('rewards')
        plt.title("rewards evaluation ml3 policy")
        plt.savefig('rewards_eval.png')
        plt.clf()

    def save_losses_plt(self,):
        plt.plot(self.task_losses)
        plt.legend(['loss'])
        plt.xlabel('ml3 update steps')
        plt.ylabel('losses')
        plt.title("losses during ml3 training")
        plt.savefig('losses_eval.png')
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





def objective(trial):
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
    }

    m = Agent(params, trial)
    loss = m.train(trial)
    m.destroy()
    return loss



def main():
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=40)
    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    optuna.visualization.plot_intermediate_values(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)

if __name__ == "__main__":
    main()
