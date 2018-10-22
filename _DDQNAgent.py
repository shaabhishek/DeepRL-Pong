import random
import torch
from _Networks import QNetworkPongCNN
from _utils import PiecewiseSchedule
from ReplayBuffer import ReplayBufferIMG
from torch.optim import Adam
import torch.nn.functional as F
import tqdm
import logging
from _DQNAgent import DQNAgent

class DDQNAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self):
        self.globalstep += 1

        cur_states, actions, rewards, next_states, dones  = self.replaybuffer.sample(self.batch_size)

        self.cur_states_b.data.copy_(torch.from_numpy(cur_states))
        self.actions_b.data.copy_(torch.from_numpy(actions))
        self.rewards_b.data.copy_(torch.from_numpy(rewards))
        self.next_states_b.data.copy_(torch.from_numpy(next_states))
        self.dones_b.data.copy_(torch.from_numpy(dones))

        best_actions = self.qnetwork(self.next_states_b).argmax(dim=1)
        next_state_q_vals = self.qnetwork_target(self.next_states_b)[range(self.batch_size), best_actions]
        target_q_vals = self.env.gamma * (1-self.dones_b) * next_state_q_vals + self.rewards_b # if done, there is no next state, so predicted_q_value should only be the reward
        target_q_vals = target_q_vals.detach()
        pred_q_vals = self.qnetwork(self.cur_states_b)[range(self.batch_size), self.actions_b.long()]

        self.optimizer.zero_grad()
        loss_val = F.mse_loss(target_q_vals, pred_q_vals)

        loss_val.backward()
        # clip_grad_norm_(self.qnetwork.parameters(), self.clip_grad_norm) #Clip grad norm
        self.optimizer.step()

        # Store Q values every self.q_log_freq train steps
        if self.globalstep%self.q_log_freq == 0:
            self.log("Step:{},QValue:{},Loss:{}".format(
                self.globalstep, round(pred_q_vals.mean().item(), 3),round(loss_val.item(),3))
            )