import numpy as np
import torch
from collections import deque
import gym
from gym import spaces
import cv2
from multiprocessing import Pipe, Process
from env_wrappers import WarpFrame, NoopResetEnv, MaxAndSkipEnv, FrameStack, ImageToPyTorch, EpisodicLifeEnv


class Pong():
    def __init__(self, *args, **kwargs):
        self.VALID_ACTION = [0,2,5]
        env = gym.make('PongNoFrameskip-v4')
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        
        env = EpisodicLifeEnv(env)
        env = WarpFrame(env)
        env = FrameStack(env, k=4)
        env = ImageToPyTorch(env)
        self.env = env
        self.gamma = kwargs['gamma']

    def step(self, action):
        obs, r, done, _ = self.env.step(self.VALID_ACTION[action])
        return torch.from_numpy(obs), r, done

    def reset(self):
        s = self.env.reset()
        return torch.from_numpy(s)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


def make_parallel_envs(**kwargs):
    num_envs = kwargs['num_envs']
    envs = [Pong(**kwargs) for _ in range(num_envs)]
    # envs = [NormalizedGymEnv(env_id, is_atari=is_atari, **kwargs)
    #         for _ in range(num_envs)]
    # [envs[i].seed(start_seed + i) for i in range(num_envs)]
    return envs


def worker(remote, parent_remote, env):
    """
    Taken from OpenAI baselines
    """
    parent_remote.close()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.env.observation_space, env.env.action_space))
        elif cmd == 'render':
            env.render()
        elif cmd == 'close':
            env.close()
        # elif cmd == 'seed':
        #     remote.send((env.seed(data)))
        # elif cmd == 'get_episode_rewards':
        #     remote.send(
        #         get_wrapper_by_name(env, 'MonitorEnv').get_episode_rewards())
        # elif cmd == 'get_total_steps':
        #     remote.send(
        #         get_wrapper_by_name(env, 'MonitorEnv').get_total_steps())
        else:
            raise NotImplementedError

class PongParallel:
    """
    Adapted from OpenAI baselines
    """

    def __init__(self, **kwargs):
        """
        :param env_id: str, environment id
        :param num_envs: int, number of environments
        :param start_seed: int, seed for environment, gets incremented by 1
            for each additional env
        """
        envs = make_parallel_envs(**kwargs)

        self.envs = envs
        self.gamma = kwargs['gamma']
        # self.start_seed = start_seed
        # self.env_id = env_id
        self.waiting = False
        self.closed = False
        self.num_envs = len(envs)
        self.parents, self.children = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [Process(target=worker, args=(child, parent, env)) for (child, parent, env) in zip(self.children, self.parents, envs)]
        for p in self.ps:
            # daemons are killed if parent is killed
            p.daemon = True
            p.start()
        for child in self.children:
            child.close()

        self.parents[0].send(('get_spaces', None))
        observation_space, action_space = self.parents[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space
        # self.spec = envs[0].spec
        # self.is_atari = is_atari

    def step_async(self, actions):
        for parent, action in zip(self.parents, actions):
            parent.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [parent.recv() for parent in self.parents]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return torch.stack(obs), torch.tensor(rews), torch.tensor(dones)

    def reset(self):
        for parent in self.parents:
            parent.send(('reset', None))
        return torch.stack([parent.recv() for parent in self.parents])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        self.parents[0].send(('render', None))

    def close(self):
        self.parents[0].send(('close', None))
    

    # def get_episode_rewards(self, last_n=None):
    #     """
    #     :param last_n: int, get the last_n rewards per env
    #     """
    #     for parent in self.parents:
    #         parent.send(('get_episode_rewards', None))
    #     results = [parent.recv() for parent in self.parents]
    #     if last_n:
    #         results = [r[-last_n:] for r in results]
    #     flat_results = []
    #     for r in results:
    #         flat_results.extend(r)
    #     return flat_results

    # def get_total_steps(self):
    #     for parent in self.parents:
    #         parent.send(('get_total_steps', None))
    #     results = [parent.recv() for parent in self.parents]
    #     return results

    # def seed(self, i):
    #     for parent in self.parents:
    #         parent.send(('seed', i))
    #         i += 1
    #     return [parent.recv() for parent in self.parents]

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for parent in self.parents:
                parent.recv()
        for parent in self.parents:
            parent.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True



def main():
    # env = Pong(gamma=0.9)
    # print(env.env.observation_space)
    # print(env.env.action_space)
    # print(env.VALID_ACTION)

    env = PongParallel(num_envs=2, gamma=0.9)
    env.reset()
    obs, rews, dones = env.step([0]*2)
    # print(env.observation_space)
    print(obs.shape, rews, dones, sep='\n')
    # print(type(obs), type(rews), type(dones))
    env.close()

if __name__ == '__main__':
    main()