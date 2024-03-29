from copy import deepcopy
from typing import List
from mlagents.torch_utils import default_device

import torch
from torch.autograd import Variable
from torch import nn, Tensor
from torch.optim import Adam

from trainer.OUNoise import OUNoise


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr):
        self.actor = MLPNetwork(obs_dim, act_dim)

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        self.critic = MLPNetwork(global_obs_dim, 1)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.exploration = OUNoise(act_dim)
        self.move_to_device(default_device())

    def move_to_device(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.target_actor.to(device)
        self.target_critic.to(device)

    def reset_noise(self):
        self.exploration.reset()

    def scale_noise(self, scale):
        self.exploration.scale = scale

    def action(self, obs, explore=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        action = self.actor(obs)  # torch.Size([batch_size, action_size])
        if explore:
            action += Variable(Tensor(self.exploration.noise()), requires_grad=False)
        action = action.clamp(-1.0, 1.0)
        # print("AGENT : local action : ", action, "\n")
        return action

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        action = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        action = action.clamp(-1.0, 1.0)
        action = action.squeeze(0).detach()
        return action

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)  # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.net(x)
