import random
import torch
from _Networks import QNetworkPongCNN
from _utils import PiecewiseSchedule
from ReplayBuffer import ReplayBufferIMG
from torch.optim import Adam
import torch.nn.functional as F
import tqdm
import logging

class DQNAgent:
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.env = kwargs['env']
        self.batch_size = kwargs['batch_size']
        self.globalstep = 0
        self.VALID_ACTION = self.env.VALID_ACTION
        self.n_actions = len(self.VALID_ACTION)
        self.lr = kwargs['lr']
        self.replay_buffer_size = kwargs["replay_buffer_size"]
        self.num_timesteps = kwargs["num_timesteps"]
        self.target_update_freq = kwargs["target_update_freq"]
        self.q_log_freq = kwargs["q_log_freq"]
        self.save_freq = kwargs["save_freq"]

        self.replaybuffer = ReplayBufferIMG(self.replay_buffer_size)
        self.epsilon_schedule = PiecewiseSchedule([(0, 1.0), (1e5, 0.1), (5e6, 0.01)], outside_value=0.01)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.qnetwork = QNetworkPongCNN(input_shape=self.env.env.observation_space.shape).to(self.device)
        self.qnetwork_target = QNetworkPongCNN(input_shape=self.env.env.observation_space.shape).to(self.device)
        self.updatetarget()
        self.optimizer = Adam(self.qnetwork.parameters(), lr=self.lr)
        self.fillreplaybuffer(kwargs["replay_buffer_prefill"])

        # for batch learning
        self.cur_states_b = torch.zeros(((self.batch_size,4,84,84))).to(self.device)
        self.next_states_b = torch.zeros(((self.batch_size,4,84,84))).to(self.device)
        self.actions_b = torch.zeros((self.batch_size,)).to(self.device)
        self.rewards_b = torch.zeros((self.batch_size,)).to(self.device)
        self.dones_b = torch.zeros((self.batch_size,)).to(self.device)

    def getepsilon(self,):
        return(self.epsilon_schedule.value(int(self.globalstep)))

    def get_action(self,state):
        # return action in reduced action-space (0-2 for Pong)

        if random.random() < self.getepsilon(): #(epsilon)-greedy
            return random.randint(0,self.n_actions-1) #random action
        else:
            state = state.to(self.device)
            # return self.qnetwork(state).max(-1)[1].item()
            return self.qnetwork(state.unsqueeze(0)).argmax().item()

    def train(self):
        self.globalstep += 1

        cur_states, actions, rewards, next_states, dones  = self.replaybuffer.sample(self.batch_size)

        self.cur_states_b.data.copy_(torch.from_numpy(cur_states))
        self.actions_b.data.copy_(torch.from_numpy(actions))
        self.rewards_b.data.copy_(torch.from_numpy(rewards))
        self.next_states_b.data.copy_(torch.from_numpy(next_states))
        self.dones_b.data.copy_(torch.from_numpy(dones))

        next_state_q_vals = self.qnetwork_target(self.next_states_b).detach().max(dim=1)[0] #greedy selection using max
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

    def updatetarget(self):
        self.qnetwork_target.load_state_dict(self.qnetwork.state_dict())

    def fillreplaybuffer(self, fill_length):
        print("Filling Replay Buffer...")
        pbar = tqdm.tqdm(total=fill_length)
        while len(self.replaybuffer) < fill_length:
            done = False
            state = self.env.reset()
            while not done:
                action=self.get_action(state)
                next_state, reward, done = self.env.step(action)
                # done = True if reward == -1 else False
                self.replaybuffer.add(state, action, reward, next_state, done)
                state = next_state
                pbar.update()
        pbar.close()

    def log(self, message):
        logging.info(message)