import multiprocessing
import multiprocessing.connection
import numpy as np
import cv2
from dmp.utils.smnist_loader import MatLoader, Separate


class SMNIST(object):
    def __init__(self,
                 max_ep_length=300,
                 goal_conditioned=False,
                 digit_idx=2,
                 randomize=False,
                 single_digit=False,
                 data_path='./dmp/data/s-mnist/40x40-smnist.mat'):

        self.data_path = data_path
        # list of digits TO-DO
        self.single_digit = single_digit
        # normalize the input image and trajectory at each time step
        self.randomize = randomize
        # time_steps until done
        self.max_ep_length = max_ep_length
        self.goal_conditioned = goal_conditioned
        self.initialise_environment_PPO(data_path, digit_idx)
        self.last_reward = 0

    @property
    def action_space(self):
        return self._action_spec

    @property
    def observation_space(self):
        return self._observation_spec

    @property
    def task_goal(self):
        # Input image
        return self._task_goal


    @property
    def all_rewards(self):
        return self.last_reward

    @property
    def goal(self):
        if self.step_no != self.max_ep_length:
            return self._target_trajectory[self.step_no + 1]
        else:
            return self._target_trajectory[self.step_no]


    def load_images(self, data_path, digit_idx):
        '''
        load all images, target trajectories and desired digit indices

        param:
            data_path: dir to the s-minst 40x40
            digit_idx: select digit from 0-9
        return:
            images: all images
            trajectories: all target trajectories
            digit_indices: indices of desired trajectories
        '''
        images, outputs, scale, trajectories = MatLoader.load_data(
            data_path, load_original_trajectories=True)
        images = np.array([cv2.resize(img, (28, 28)) for img in images]) / 255.0
        data_sep = Separate()
        digit_indices = data_sep.no_separation()
        return images, trajectories, digit_indices[digit_idx].astype(int)

    def initialise_environment_PPO(self, path, digit_idx):
        '''
        Intialize the environment for a single action step.

        param:
            path: dir to s-mnist 40x40
            digit_idx: select digit from 0-9
        return:
            None
        '''
        self._images, self._trajectories, self._desired_idx = self.load_images(
            self.data_path, digit_idx)

        self._task_goal = np.zeros_like(self._images[0])
        self._action_spec = np.zeros(2)
        self._state = np.zeros(np.array(self._trajectories)[:, :, :2].shape[-1])

        if not self.goal_conditioned:
            # Input of dimension 2 "shape of startPos"
            self._observation_spec = np.zeros_like(self._state)
        else:
            # Input includes startPose and shape of target image
            total_dim = self._task_goal.reshape([-1]).shape[0] + self._state.reshape([-1]).shape[0]
            self._observation_spec = np.zeros((total_dim,))

    def reset(self, perturb_goal=False):

        self.accum_rewards = []

        if self.randomize:
            np.random.shuffle(self._desired_idx)

        # Input image and target trajectory
        self._task_goal = self._images[self._desired_idx[0]].astype(np.float32)
        self._target_trajectory = self.extract_demo()

        # start position
        self._state = self._target_trajectory[0, :]
        self.state_visited = [self._state]

        self._episode_ended = False
        self.step_no = 0

        return self._state

    def extract_demo(self):
        '''
        extract trajectory of the desired_idx

        return:
            trajectory that correspond to the target trajectory
        '''
        return np.array(self._trajectories)[:, :, :2].astype(np.float32)[self._desired_idx[0]]

    def commute_reward(self, states_list):
        # TO-DO
        count = 0
        xs = []
        ys = []
        goal = self._task_goal
        pred_image = np.zeros_like(self._task_goal)

        for i in range(len(states_list)):
            x_axis = np.clip(round(states_list[i][1] * 0.668), 0, 27)
            y_axis = np.clip(round(states_list[i][0] * 0.668), 0, 27)
            pred_image[x_axis, y_axis] = 1.0 / 255.0

            check = x_axis in xs and y_axis in ys

            if not check:
                if goal[x_axis, y_axis] * 255.0 > 0.3:
                    count += 1
                    xs.append(x_axis)
                    ys.append(y_axis)
                else:
                    count -= 1
            else:
                count -= 1

            # distance = (np.abs(goal - pred_image))
            # reward = np.exp(-np.mean(distance))
            # print("rewards: ", distance.sum())
        return count

    def step(self, action):
        self.step_no += 1
        # Make sure episodes don't go on forever.
        if self.step_no == self.max_ep_length:
            self.last_reward = sum(self.accum_rewards)
            self.reset()
            return self._state, 0.0, True, self.last_reward

        self._state = self._state + action
        self._goal = self._target_trajectory[self.step_no]
        distance = np.abs(self._goal-self._state)
        reward = np.exp(-np.mean(distance))
        self.accum_rewards.append(reward)
        return self._state, reward, False, None


def worker_process(remote: multiprocessing.connection.Connection, seed: int):
    game = SMNIST()
    while True:
        cmd, data= remote.recv()
        if cmd == "step":
            remote.send(game.step(data))
        elif cmd == "reset":
            remote.send(game.reset())
        elif cmd == "goal":
            remote.send(game.goal)
        elif cmd == "reward":
            remote.send(game.last_reward)
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError

class Worker:
    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, seed))
        self.process.start()


