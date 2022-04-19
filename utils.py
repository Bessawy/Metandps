import numpy as np
import torch
import matplotlib.pyplot as plt
from visdom import Visdom
from smnistenv2 import Worker


def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)


def pre_process(samples, device):
    samples_flat = {}
    for k, v in samples.items():
        v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
        samples_flat[k] = obs_to_torch(v, device)
    return samples_flat


def to_torch(samples, device):
    samples_torch = {}
    for k, v in samples.items():
        samples_torch[k] = obs_to_torch(v, device)
    return samples_torch


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print("device: ", device)
    return device

def discount_rewards(rewards, gamma=0.99):
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def save_Tasklosses_plt(loss, inner_iter, range, name):
    plt.plot(loss)
    plt.legend(['Task losses'])
    plt.xlabel('ml3 updates')
    plt.ylabel('losses')
    plt.ylim(range)
    plt.title("Task losses during ml3 training for inner: " + str(iter))
    plt.savefig(name + '_losses.png')
    plt.clf()

def store_model(model, name):
    torch.save(model.state_dict(), name + '_model.mdl')

def load_model(model, path, device):
    weights = torch.load(path, map_location=device)
    model.load_state_dict(weights, strict=False)
    return "model_loaded"

def show_trajectory(task_goal, trajecx, trajecy, name):
    fig, ax = plt.subplots()
    im = ax.imshow(task_goal)
    ax.plot(np.array(trajecx) * 0.668, np.array(trajecy) * 0.668, 'x', color='red')
    plt.savefig(name + 'digit.png')
    plt.clf()
    plt.close(fig)

class VisdomLinePlotter(object):
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

