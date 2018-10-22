# First file: dqn_0.19446710549615254 (timesteps: 400000) lr=1e-4
# Second file: dqn_0.4036237050247964 (timesteps: 400000) lr=5e-5
# Third file: dqn_0.8246659563481628 (timesteps: 400000) previous optimizer state
# Fourth file: dqn_0.6389490337149797 (timesteps: 400000) lr=1e-4, replaybuffersize = 1000000 -> 100000, target_update_freq= 5000 -> 1000


from _PongEnv import Pong
from _DQNAgent import DQNAgent
from _DDQNAgent import DDQNAgent

import numpy as np
import torch
import logging
from collections import deque
from _utils import savemodel, loadmodel
import random
import time

def eval(agent, env, render=False):
    done = False
    state = env.reset()
    ep_rewards = 0
    t = 0
    while True:
        if render:
            time.sleep(0.02)
            env.render()
        t += 1
        action = agent.get_action(state) # act expects state as a tensor
        next_state, reward, done = env.step(action)
        ep_rewards += reward
        state = next_state
        if done or t > 50000:
            break
    env.close()
    if render: env.close()
    return ep_rewards, t

def run_dqn(agent, env):
    
    filename = 'dqn_'+str(random.random())
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

            # Save the model
            if agent.globalstep % agent.save_freq == 0:
                savemodel(agent.qnetwork, 'models/'+filename+'-qmodel'+str(agent.globalstep))
                savemodel(agent.optimizer, 'models/'+filename+'-qmodeloptimizer'+str(agent.globalstep))

        print("Reward After Episode {}: {}".format(episode, ep_reward))
        logging.info("Episode:{},Reward:{},EpisodeLength:{}".format(int(episode), int(ep_reward), t))
        ep_rewards.append(ep_reward)

        # Termination condition
        if agent.globalstep > 3000000:
            keep_learning = False

    np.save(filename, np.array(ep_rewards))
    savemodel(agent.qnetwork, 'models/'+filename+'-qmodel'+str(episode))
    # return episode


def main():
    params_dict_env = {
        "gamma": 0.99,
    }
    env = Pong(**params_dict_env)

    def get_params_agent(train=True):
        # params_dict_agent = {
        #     "env": env,
        #     # "batch_size": 32,
        #     "batch_size": 64,
        #     # "replay_buffer_size": 100000,
        #     "replay_buffer_size": 1000000,
        #     # "replay_buffer_prefill": 10000,
        #     # "replay_buffer_prefill": 1,
        #     "lr": 5e-5,
        #     # "lr": 1e-4,
        #     "num_timesteps": 2000000,
        #     # "target_update_freq": 1000,
        #     "target_update_freq": 5000,
        #     # logging
        #     "save_freq":50000,
        #     "q_log_freq": 1000
        # }
        # Original params
        params_dict_agent = {
            "env": env,
            "batch_size": 32,
            "replay_buffer_size": 100000,
            "replay_buffer_prefill": 10000,
            "lr": 1e-4,
            "num_timesteps": 3000000,
            "target_update_freq": 1000,
            # logging
            "save_freq":50000,
            "q_log_freq": 1000
        }
        if train:
            params_dict_agent["replay_buffer_prefill"]= 10000
        else:
            params_dict_agent["replay_buffer_prefill"]= 1
        return params_dict_agent

    train=False
    params_dict_agent = get_params_agent(train)

    agent = DQNAgent(**params_dict_agent)
    
    preload_model_filename = "dqn_0.47356275336026943"
    step = 3000000
    loadmodel(agent.qnetwork, 'models/'+preload_model_filename+'-qmodel'+str(step), agent.device.type)
    print("Loading model: {}".format(('models/'+preload_model_filename+'-qmodel'+str(step))) )

    if train:
        loadmodel(agent.optimizer, 'models/'+preload_model_filename+'-qmodeloptimizer'+str(step), agent.device.type)
        print("Loading optimizer: {}".format('models/'+preload_model_filename+'-qmodeloptimizer'+str(step)) )
    agent.updatetarget()
    agent.globalstep = step

    if train:
        # Training
        run_dqn(agent, env)
    else:
        # For playing
        print(eval(agent, env, render=True))

if __name__ == '__main__':
    main()


