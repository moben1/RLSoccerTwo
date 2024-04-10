import threading
from random import sample

import torch
from numpy import ndarray
from torch import Tensor

LOCK_MEMORY = threading.Lock()

config = {
    "epochs": 100000,
    "evaluate_epoch" : 1,
    "show_evaluate_epoch" : 20,
    "memory_batch" : 32,
    "memory_size" : 1000,
    "run_episode_before_train" : 3,  # Run several episodes with the same strategy, used in on-policy algorithms
    "learn_num" : 2,
    "lr_actor" : 1e-4,
    "lr_critic" : 1e-3,
    "gamma" : 0.99,  # reward discount factor
    "epsilon" : 0.7,
    "grad_norm_clip" : 10,
    "target_update_cycle" : 100,
    "save_epoch" : 1000,
    "model_dir" : r"./models",
    "result_dir" : r"./results",
    "cuda" : True,
}


class CommBatchEpisodeMemory(object):
    """
        Memory cells for storing information of each game, suitable for conventional marl algorithms 
    """

    def __init__(self, continuous_actions: bool, n_actions: int = 0, n_agents: int = 0):
        self.continuous_actions = continuous_actions
        self.n_actions = n_actions
        self.num_agents = n_agents
        self.obs_buffs = []
        self.obs_next = []
        self.state = []
        self.state_next = []
        self.rewards = []
        self.unit_actions = []
        self.log_probs = []
        self.unit_actions_onehot = []
        self.per_episode_len = []
        self.n_step = 0

    def store_one_episode(self, one_obs: dict, one_state: ndarray, action: list, reward: float,
                          one_obs_next: dict = None, one_state_next: ndarray = None, log_probs: list = None):
        one_obs = torch.stack([torch.Tensor(value) for value in one_obs.values()], dim=0)
        self.obs_buffs.append(one_obs)
        self.state.append(torch.Tensor(one_state))
        self.rewards.append(reward)
        self.unit_actions.append(action)
        if one_obs_next is not None:
            one_obs_next = torch.stack([torch.Tensor(value) for value in one_obs_next.values()], dim=0)
            self.obs_next.append(one_obs_next)
        if one_state_next is not None:
            self.state_next.append(torch.Tensor(one_state_next))
        if log_probs is not None:
            self.log_probs.append(log_probs)
        if not self.continuous_actions:
            self.unit_actions_onehot.append(
                torch.zeros(self.num_agents, self.n_actions).scatter_(1, torch.LongTensor(action).unsqueeze(dim=-1), 1))
        self.n_step += 1

    def clear_memories(self):
        self.obs_buffs.clear()
        self.obs_next.clear()
        self.state.clear()
        self.state_next.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.unit_actions.clear()
        self.unit_actions_onehot.clear()
        self.per_episode_len.clear()
        self.n_step = 0

    def set_per_episode_len(self, episode_len: int):
        self.per_episode_len.append(episode_len)

    def get_batch_data(self) -> dict:
        """
        Fetch a batch of data
        :return: A batch of data encapsulated in a dictionary
        """
        obs = torch.stack(self.obs_buffs, dim=0)
        state = torch.stack(self.state, dim=0)
        rewards = reshape_tensor_from_list(torch.Tensor(self.rewards), self.per_episode_len)
        actions = torch.Tensor(self.unit_actions)
        log_probs = torch.Tensor(self.log_probs)
        data = {
            'obs': obs,
            'state': state,
            'rewards': rewards,
            'actions': actions,
            'log_probs': log_probs,
            'per_episode_len': self.per_episode_len
        }
        return data


class CommMemory(object):
    """
        Memory cells for storing information of all games, suitable for conventional MARL algorithms
    """

    def __init__(self):
        self.train_config = config
        self.memory_size = self.train_config["memory_size"]
        self.current_idx = 0
        self.memory = []

    def store_episode(self, one_episode_memory: CommBatchEpisodeMemory):
        with LOCK_MEMORY:
            obs = torch.stack(one_episode_memory.obs_buffs, dim=0)
            obs_next = torch.stack(one_episode_memory.obs_next, dim=0)
            state = torch.stack(one_episode_memory.state, dim=0)
            state_next = torch.stack(one_episode_memory.state_next, dim=0)
            actions = torch.Tensor(one_episode_memory.unit_actions)
            actions_onehot = torch.stack(one_episode_memory.unit_actions_onehot, dim=0)
            reward = torch.Tensor(one_episode_memory.rewards)
            data = {
                'obs': obs,
                'obs_next': obs_next,
                'state': state,
                'state_next': state_next,
                'rewards': reward,
                'actions': actions,
                'actions_onehot': actions_onehot,
                'n_step': one_episode_memory.n_step
            }
            if len(self.memory) < self.memory_size:
                self.memory.append(data)
            else:
                self.memory[self.current_idx % self.memory_size] = data
            self.current_idx += 1

    def sample(self, batch_size) -> dict:
        """
        Randomly sample from the memory cells, but the number of steps in each game is different. Find the largest number of steps in this batch, and pad the data for other games to match.
        :param batch_size: The size of a batch
        :return: A batch of data
        """
        sample_size = min(len(self.memory), batch_size)
        sample_list = sample(self.memory, sample_size)
        n_step = torch.Tensor([one_data['n_step'] for one_data in sample_list])
        max_step = int(torch.max(n_step))

        obs = torch.stack(
            [torch.cat([one_data['obs'],
                        torch.zeros([max_step - one_data['obs'].shape[0]]
                                    + list(one_data['obs'].shape[1:]))
                        ])
             for one_data in sample_list], dim=0).detach()

        obs_next = torch.stack(
            [torch.cat([one_data['obs_next'],
                        torch.zeros(size=[max_step - one_data['obs_next'].shape[0]] +
                                    list(one_data['obs_next'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        state = torch.stack(
            [torch.cat([one_data['state'],
                        torch.zeros([max_step - one_data['state'].shape[0]] +
                                    list(one_data['state'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        state_next = torch.stack(
            [torch.cat([one_data['state_next'],
                        torch.zeros([max_step - one_data['state_next'].shape[0]] +
                                    list(one_data['state_next'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        rewards = torch.stack(
            [torch.cat([one_data['rewards'],
                        torch.zeros([max_step - one_data['rewards'].shape[0]] +
                                    list(one_data['rewards'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        actions = torch.stack(
            [torch.cat([one_data['actions'],
                        torch.zeros([max_step - one_data['actions'].shape[0]] +
                                    list(one_data['actions'].shape[1:]))])
             for one_data in sample_list], dim=0).unsqueeze(dim=-1).detach()

        actions_onehot = torch.stack(
            [torch.cat([one_data['actions_onehot'],
                        torch.zeros([max_step - one_data['actions_onehot'].shape[0]] +
                                    list(one_data['actions_onehot'].shape[1:]))])
             for one_data in sample_list], dim=0).detach()

        terminated = torch.stack(
            [torch.cat([torch.ones(one_data['n_step']), torch.zeros(max_step - one_data['n_step'])])
             for one_data in sample_list], dim=0).detach()

        batch_data = {
            'obs': obs,
            'obs_next': obs_next,
            'state': state,
            'state_next': state_next,
            'rewards': rewards,
            'actions': actions,
            'actions_onehot': actions_onehot,
            'max_step': max_step,
            'sample_size': sample_size,
            'terminated': terminated
        }
        return batch_data

    def get_memory_real_size(self):
        return len(self.memory)


def reshape_tensor_from_list(tensor: Tensor, shape_list: list) -> list:
    """
    Split the tensor according to the shape_list, to address the issue of different game lengths within a batch (this might not be an issue in PettingZoo, but it exists in SMAC), by placing the tensor into a list.
    :param tensor: The input tensor
    :param shape_list: A list of lengths for each game
    :return: The tensor split according to the lengths of each game, with the results encapsulated in a list
    """
    if len(tensor) != sum(shape_list):
        raise ValueError("value error: len(tensor.shape) not equals sum(shape_list)")
    if len(tensor.shape) != 1:
        raise ValueError("value error: len(tensor.shape) != 1")
    rewards = []
    current_index = 0
    for i in shape_list:
        rewards.append(tensor[current_index:current_index + i])
        current_index += i
    return rewards
