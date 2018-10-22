"""
A2C - synchronous advantage actor critic with generalized advantage estimation

Adapted from https://arxiv.org/pdf/1602.01783.pdf, Algorithm S3

"""
# import time
import torch
import numpy as np
from _Networks import A2CNetworkPongCNN
import torch.nn.functional as F
from torch.optim import RMSprop
import torch.nn.utils
from _utils import discount_with_dones
# from yarlp.agent.base_agent import Agent, add_advantage
# from yarlp.model.networks import cnn
# from yarlp.utils.experiment_utils import get_network
# from yarlp.model.model_factories import a2c_model_factory
# from yarlp.utils.env_utils import ParallelEnvs
# from yarlp.utils.metric_logger import explained_variance
# from yarlp.utils.schedules import PiecewiseSchedule, ConstantSchedule
# from dateutil.relativedelta import relativedelta as rd


class A2CAgent:

    # def __init__(
    #         self, env,
    #         policy_network=None,
    #         policy_network_params={
    #             'final_dense_weights_initializer': 0.01
    #         },
    #         policy_learning_rate=5e-4,
    #         value_network_params={
    #             'final_dense_weights_initializer': 1.0
    #         },
    #         entropy_weight=0.01,
    #         model_file_path=None,
    #         adaptive_std=False, init_std=1.0, min_std=1e-6,
    #         n_steps=5,
    #         max_timesteps=1000000,
    #         grad_norm_clipping=0.5,
    #         gae_lambda=0.98,
    #         checkpoint_freq=10000,
    #         save_freq=50000,
    #         policy_learning_rate_schedule=None,
    #         *args, **kwargs):

        # super().__init__(env, *args, **kwargs)

        # assert isinstance(self._env, ParallelEnvs),\
        #     "env must be ParallelEnvs class for A2C agent"
    def __init__(self, *args, **kwargs):
        self.env = kwargs['env']

        # if torch.cuda.is_available():
        #     self.device = torch.device('cuda')
        # else:
        self.device = torch.device('cpu')

        self.entropy_weight = kwargs["entropy_weight"]
        self.vf_loss_weight = kwargs["vf_loss_weight"]
        self.lr = kwargs["lr"]
        self.lr_epsilon = kwargs["lr_epsilon"]
        self.lr_decay = kwargs["lr_decay"]
        self.max_grad_norm = kwargs["max_grad_norm"]
        self.num_steps = kwargs["num_steps"]
        self.checkpoint_freq = kwargs["checkpoint_freq"]
        self.save_freq = kwargs["save_freq"]
        self.max_timesteps = kwargs["max_timesteps"]
        self.t = 0

        self.policy = A2CNetworkPongCNN(input_shape=self.env.observation_space.shape).to(self.device)
        self.optimizer = RMSprop(self.policy.parameters(), lr=self.lr, eps=self.lr_epsilon,  alpha=self.lr_decay)

        self.states = self.env.reset().to(self.device)

    def get_batch_actions(self, batch_obs, greedy=False):
        action_logprobs, values = self.policy(batch_obs)
        if greedy:
            actions = action_logprobs.argmax(-1)
        else:
            actions = torch.distributions.Categorical(logits=action_logprobs).sample()
        return actions, values.squeeze()

    def run(self, n_steps=5):
        #say (100,4,84,84)
        self.t += self.env.num_envs * n_steps
        batch_ob_shape = (self.env.num_envs * n_steps, *self.env.observation_space.shape)
        mb_obs, mb_rewards, mb_actions = [], [], []
        mb_values, mb_dones = [], []

        for n in range(n_steps):
            actions, values = self.get_batch_actions(self.states, greedy=False)
            mb_obs.append(self.states)
            mb_actions.append(actions)
            mb_values.append(values)
            next_states, rewards, dones = self.env.step(actions)
            mb_dones.append(dones)
            # mb_dones.append(np.array(dones))
            mb_rewards.append(rewards)

            self.states = next_states
        
        mb_obs = torch.stack(mb_obs).transpose(1, 0).reshape(batch_ob_shape)
        mb_rewards = torch.stack(mb_rewards).transpose(1, 0)
        mb_actions = torch.stack(mb_actions).transpose(1, 0)
        mb_values = torch.stack(mb_values).transpose(1, 0)
        mb_dones = torch.stack(mb_dones).transpose(1, 0)

        _, last_values = self.get_batch_actions(self.states)

        mb_discounted_rewards = []

        # for n in range(self.env.num_envs):
        #     rewards = mb_rewards[n]
        #     dones = mb_dones[n]
        #     last_value = last_values[n]
        #     # If the episode ended at last step, return of last step is just the reward at last step
        #     # Else, return of last step is bootstrapped off its next state's state-value
        #     discounted_rewards = self.discount_with_dones(torch.cat(( rewards, last_value.unsqueeze(0) )), dones, self.env.gamma)
        #     mb_discounted_rewards.append(discounted_rewards)

        for n in range(self.env.num_envs):
            rollout = {
                "baseline_preds": mb_values[n].flatten(),
                "next_baseline_pred": last_values[n].flatten(),
                "dones": mb_dones[n],
                "rewards": mb_rewards[n]
            }
            self.add_advantage(rollout, self.env.gamma)
            mb_discounted_rewards.append(rollout["discounted_rewards"])

        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_discounted_rewards = torch.stack(mb_discounted_rewards).flatten()

        return mb_obs, mb_actions, mb_discounted_rewards, mb_values

    # len(dones) = t
    # len(rewards_and_value) = t+1
    def discount_with_dones(self, rewards_and_value, dones, gamma):
        T = len(rewards_and_value)
        # rewards_and_value is actually [rewards[0:(t+1)], value_t+1]
        returns = torch.zeros_like(rewards_and_value)
        
        # If the episode ended at last step, set the value function to be G_t+1
        if dones[-1] == 0:
            returns[-1] = rewards_and_value[-1]
        for t in reversed(range(T-1)):
            isterminal = dones[t].float() #dones = 0 implies terminal = 0
            returns[t] = rewards_and_value[t] + gamma * returns[t+1] * (1-isterminal)
        # G_t+1 was actually the value we bootstrap G_t off, and we don't need it
        return returns[:-1]

    def add_advantage(self, rollout, gamma):
        # baseline_preds = np.append(rollout["baseline_preds"], rollout["next_baseline_pred"])
        baseline_preds = torch.cat((rollout["baseline_preds"], rollout["next_baseline_pred"].unsqueeze(0)))
        T = len(rollout["rewards"])
        rollout["advantages"] = gaelam = torch.zeros_like(rollout["rewards"])
        rollout["discounted_rewards"] = torch.zeros_like(rollout["rewards"])
        rewards = rollout["rewards"]
        lastgaelam = 0
        r = rollout["next_baseline_pred"] if not rollout["dones"][-1] else 0
        for t in reversed(range(T)):
            nonterminal = (1 - rollout["dones"][t]).float()
            rollout["discounted_rewards"][t] = r = rewards[t] + gamma * r * nonterminal
            delta = rewards[t] + gamma * baseline_preds[t + 1] * nonterminal - baseline_preds[t]
            gaelam[t] = lastgaelam = delta + gamma * nonterminal * lastgaelam

        rollout["discounted_future_reward"] = rollout["advantages"] + rollout["baseline_preds"]

    def train(self):
        states, actions, returns, values = self.run(self.num_steps)

        # get prediction logprobs probabilities
        pred_action_logprobs, pred_values = self.policy(states)
        
        # cast logprobs as a PyTorch Categorical Distribution for log_prob and entropy convenience functions
        policy_predictions_distribution = torch.distributions.Categorical(logits=pred_action_logprobs)
        
        # get log probs for actions that are taken and negate them
        neglogp_actions = -policy_predictions_distribution.log_prob(actions)

        # compute advantages
        advantages = (returns - values).detach()

        # pg-loss
        pg_loss = (advantages * neglogp_actions).mean()

        # value-loss
        val_loss =  self.vf_loss_weight * F.mse_loss(returns, pred_values.squeeze())

        # entropy loss. we want to minimize losses, and increase entropy (ensures exploration)
        entropy = policy_predictions_distribution.entropy().mean()
        entropy_loss = self.entropy_weight * entropy

        loss = pg_loss - entropy_loss + val_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        ev = 0 if returns.var() == 0 else 1 - ((returns - values).var() / returns.var()).abs()
        return [round(pg_loss.item(), 4), round(val_loss.item(), 4), round(entropy.item(), 4), round(ev.item(), 2)]