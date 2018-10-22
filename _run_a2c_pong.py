# First file: a2c_0.6565708838934593 ("max_timesteps": 1200000) lr=1e-4
# Second file: a2c_0.5250580183209552 ("max_timesteps": 2400000) lr=1e-4
# Third file: a2c_0.291884179427754 ("max_timesteps": 1200000) lr=1e-5
# Fourth file: a2c_0.5333673473209196 ("max_timesteps": 1200000) lr=1e-5
# Fifth file: a2c_0.8323747714184345 ("max_timesteps": 1200000) lr=5e-5
# Sixth file: a2c_0.839309354318 ("max_timesteps": 1200000) lr=5e-5
# Seventh file: a2c_0.38985953554965636 ("max_timesteps": 1200000) entropy coeff = 0.01-> 0 
# Solved..

from _PongEnv import PongParallel
from _A2CAgent import A2CAgent

import numpy as np
import torch
import logging
from collections import deque
from _utils import savemodel, loadmodel
import random
import time

def eval(agent, render=False):
    params_dict_env = {
        "gamma": 0.99,
        "num_envs": 1
    }
    env = PongParallel(**params_dict_env)
    done = False
    state = env.reset()
    ep_rewards = 0
    t = 0
    while True:
        if render:
            time.sleep(0.02)
            env.render()
        t += 1
        actions,_ = agent.get_batch_actions(state, greedy=True) # act expects state as a tensor
        next_state, reward, done = env.step(actions)
        ep_rewards += reward.item()
        state = next_state
        if done.item() or t > 50000:
            break
    env.close()
    if render: env.close()
    return ep_rewards, t

def run_a2c(agent, render=False, filename=None):
    
    # if filename:
    #     logging.basicConfig(filename='logs/'+filename+'-log', level=logging.DEBUG)
    # else:
    #     filename = 'a2c_'+str(random.random())
    #     logging.basicConfig(filename='logs/'+filename+'-log', level=logging.DEBUG)
    
    filename = 'a2c_'+str(random.random())
    logging.basicConfig(filename='logs/'+filename+'-log', level=logging.DEBUG)
    # ep_rewards = []
    i = 0
    while agent.t < agent.max_timesteps:
    # while i < 21:
        i+=1
        pg_loss, val_loss, entropy, ev = agent.train()
        
        # Log statistics
        if agent.t % agent.checkpoint_freq == 0:
        # if agent.t % 200 == 0:
            ep_reward, ep_length = eval(agent)
            print("Time:{},Rew:{},EpLength:{}".format(agent.t, ep_reward, ep_length))
            print("Iter:{},PGLoss:{},ValLoss:{},Entropy:{},ExplainedVar:{}".format(i, pg_loss, val_loss, entropy, ev))
            logging.info("Time:{},Rew:{},EpLength:{}".format(agent.t, ep_reward, ep_length))
            logging.info("Iter:{},PGLoss:{},ValLoss:{},Entropy:{},ExplainedVar:{}".format(i, pg_loss, val_loss, entropy, ev))

        # Save model
        if agent.t % agent.save_freq == 0:
            savemodel(agent.policy, 'models/'+filename+'-policymodel'+str(agent.t))
            savemodel(agent.optimizer, 'models/'+filename+'-policymodeloptimizer'+str(agent.t))
    savemodel(agent.policy, 'models/'+filename+'-policymodel'+str(agent.t))
    savemodel(agent.optimizer, 'models/'+filename+'-policymodeloptimizer'+str(agent.t))





def main():
    params_dict_env = {
        "gamma": 0.99,
        "num_envs": 24,
        # "num_envs": 1
    }
    env = PongParallel(**params_dict_env)

    params_dict_agent = {
        "env": env,
        "entropy_weight": 0.001,
        "vf_loss_weight": 0.5,
        "lr": 1e-5,
        "lr_epsilon": 1e-5,
        "lr_decay": 0.99,
        "max_grad_norm": 1,
        "num_steps": 5,
        "max_timesteps": 1200000,
        # Logging
        "checkpoint_freq":12000,
        "save_freq":60000,
        # "batch_size": 32,
        # "replay_buffer_size": 100000,
        # "replay_buffer_prefill": 10000,
        # "num_timesteps": 1000000,
        # "target_update_freq": 1000,
        # # logging
        # "q_log_freq": 500
    }

    train = False
    if train:
        params_dict_env = {
            "gamma": 0.99,
            "num_envs": 24,
        }
    else:
        params_dict_env = {
            "gamma": 0.99,
            "num_envs": 1
        }

    env = PongParallel(**params_dict_env)

    params_dict_agent = {
        "env": env,
        "entropy_weight": 0,
        "vf_loss_weight": 0.5,
        "lr": 5e-5,
        "lr_epsilon": 1e-5,
        "lr_decay": 0.99,
        "max_grad_norm": 1,
        "num_steps": 5,
        "max_timesteps": 1200000,
        # Logging
        "checkpoint_freq":12000,
        "save_freq":60000,
    }
    agent = A2CAgent(**params_dict_agent)
    preload_model_filename = "a2c_0.839309354318"
    loadmodel(agent.policy, 'models/'+preload_model_filename+'-policymodel1200000', agent.device.type)
    if train:
        loadmodel(agent.optimizer, 'models/'+preload_model_filename+'-policymodeloptimizer1200000', agent.device.type)

    if train:
        # For training
        run_a2c(agent)
    else:
        # For playing
        print(eval(agent, render=True))

    env.close()

if __name__ == '__main__':
    main()
