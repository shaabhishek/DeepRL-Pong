from _PongEnv import Pong
from _DQNAgent import DQNAgent
from _DDQNAgent import DDQNAgent

import numpy as np
import torch
import gym
import logging
from collections import deque
from _utils import savemodel
import random

def run_ddqn(agent, render=False):
    
    filename = 'ddqn_'+str(random.random())
    logging.basicConfig(filename='logs/'+filename+'-log', level=logging.DEBUG)
    ep_rewards = []
    episode = 0
    keep_learning = True
    while keep_learning:
        episode += 1
        ep_reward = 0
        state = env.reset()
        done = False
        t=0
        while not done:
            action = agent.get_action(state) # act expects state as a tensor
            next_state, reward, done = env.step(action)
            # done = True if reward == -1 else False
            agent.replaybuffer.add(state, action, reward, next_state, done)
            ep_reward += reward
            t += 1
            state = next_state
            agent.train()

            # Time to update the target network
            if agent.globalstep % agent.target_update_freq == 0:
                agent.updatetarget()

        logging.info("Episode:{},Reward:{},EpisodeLength:{}".format(int(episode), int(ep_reward), t))
        ep_rewards.append(ep_reward)

        if episode%1 == 0:
            print("Reward After Episode {}: {}".format(episode, ep_reward))
        
        # Save model
        if episode%500 == 0:
            savemodel(agent.qnetwork, 'models/'+filename+'-qmodel'+str(episode))
        
        # Termination condition
        if agent.globalstep > agent.num_timesteps:
            keep_learning = False

    np.save(filename, np.array(ep_rewards))
    savemodel(agent.qnetwork, 'models/'+filename+'-qmodel'+str(episode))
    return episode

params_dict_env = {
    "gamma": 0.99,
}
env = Pong(**params_dict_env)

params_dict_agent = {
    "env": env,
    "batch_size": 32,
    "replay_buffer_size": 1000000,
    "replay_buffer_prefill": 50000,
    "lr": 0.00025,
    "num_timesteps": 1000000,
    "target_update_freq": 1000,
    # logging
    "q_log_freq": 500
}
agent = DDQNAgent(**params_dict_agent)

run_ddqn(agent)
print(agent.get_action(env.reset()))