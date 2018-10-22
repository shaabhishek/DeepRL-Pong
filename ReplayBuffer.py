import torch
import random
import numpy as np

from collections import deque

class ReplayBuffer:
    def __init__(self, size):
        self.maxlen = size
        # self.memory_curstate = deque(maxlen=size)
        # self.memory_reward = deque(maxlen=size)
        # self.memory_nextstate = deque(maxlen=size)
        # self.memory_done = deque(maxlen=size)
        self.memory = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        # idxes: state:0, action:1, reward:2, next_state:3, done:4
        self.memory.append((state,
                            action,
                            reward,
                            next_state,
                            done))
        # self.memory_curstate.append()
        # self.memory_reward.append()
        # self.memory_nextstate.append()
        # self.memory_done.append()

    def sample(self, batch_size):
        rand_sample = random.sample(self.memory, batch_size)
        cur_states = torch.cat([trans[0] for trans in rand_sample])
        actions = torch.LongTensor([trans[1] for trans in rand_sample])
        rewards = torch.FloatTensor([trans[2] for trans in rand_sample])
        next_states = torch.cat([trans[3] for trans in rand_sample])
        dones = torch.FloatTensor([trans[4] for trans in rand_sample])

        return cur_states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    

class ReplayBufferIMG(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones, dtype=np.float32)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

# class ReplayBufferIMG(object):

#     def __init__(self,
#                  max_size=10000,
#                  bs=64,
#                  im_size=84,
#                  stack=4):

#         self.s = np.zeros((max_size, stack+1, im_size, im_size), dtype=np.float32)
#         self.r = np.zeros(max_size, dtype=np.float32)
#         self.a = np.zeros(max_size, dtype=np.int32)
#         #self.ss = np.zeros_like(self.s)
#         self.done = np.array([False]*max_size)

#         self.max_size = max_size
#         self.bs = bs
#         self._cursor = None
#         self.total_idx = list(range(self.max_size))


#     def put(self, sars):

#         if self._cursor == (self.max_size-1) or self._cursor is None :
#             self._cursor = 0
#         else:
#             self._cursor += 1

#         self.s[self._cursor] = sars[0]
#         self.a[self._cursor] = sars[1]
#         self.r[self._cursor] = sars[2]
#         #self.ss[self._cursor] = sars[3]
#         self.done[self._cursor] = sars[3]


#     def batch(self):

#         sample_idx = random.sample(self.total_idx, self.bs)
#         s = self.s[sample_idx, :4]
#         a = self.a[sample_idx]
#         r = self.r[sample_idx]
#         #ss = self.ss[sample_idx]
#         ss = self.s[sample_idx, 1:]
#         done = self.done[sample_idx]

#         return (s, a.astype(np.float32), r, ss, done.astype(np.float32))
    
#     def add(self, state, action, reward, next_state, done):
#         self.put((next_state,action,reward,done)) #next_state is 5 frames, and therefore encodes both state and next_state

#     def sample(self):
#         return self.batch()

#     def __len__(self):
#         return 0 if self._cursor is None else self._cursor