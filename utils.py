import numpy as np
import torch
import matplotlib.pyplot as plt
from visdom import Visdom


def obs_to_torch(obs: np.ndarray, device):
    return torch.tensor(obs, dtype=torch.float32, device=device)

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