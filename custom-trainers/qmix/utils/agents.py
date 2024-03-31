import random

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal

from algorithms.qmix import QMix
from utils.config_utils import ConfigObjectFactory


class QmixAgent:
    def __init__(self, env_info):
        self.env_info = env_info
        self.train_config = ConfigObjectFactory.get_train_config()
        self.env_config = ConfigObjectFactory.get_environment_config()
        self.n_agents = len(self.env_info.agents)

        if self.train_config.cuda:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.n_actions = self.env_info.action_spaces[self.env_info.agents[0]].n
        self.policy = QMix(self.env_info)

    def learn(self, batch_data: dict, episode_num: int = 0):
        self.policy.learn(batch_data, episode_num)

    def choose_actions(self, obs: dict) -> tuple:
        actions_with_name = {}
        actions = []
        log_probs = []
        obs = torch.stack([torch.Tensor(value) for value in obs.values()], dim=0)
        self.policy.init_hidden(1)
        actions_ind = [i for i in range(self.n_actions)]
        for i, agent in enumerate(self.env_info['agents_name']):
            inputs = list()
            inputs.append(obs[i, :])
            inputs.append(torch.zeros(self.n_actions))
            agent_id = torch.zeros(self.n_agents)
            agent_id[i] = 1
            inputs.append(agent_id)
            inputs = torch.cat(inputs).unsqueeze(dim=0).to(self.device)
            with torch.no_grad():
                hidden_state = self.policy.eval_hidden[:, i, :]
                q_value, _ = self.policy.rnn_eval(inputs, hidden_state)
            if random.uniform(0, 1) > self.train_config.epsilon:
                action = random.sample(actions_ind, 1)[0]
            else:
                action = int(torch.argmax(q_value.squeeze()))
            actions_with_name[agent] = action
            actions.append(action)
        return actions_with_name, actions, log_probs

    def save_model(self):
        self.policy.save_model()

    def load_model(self):
        self.policy.load_model()

    def del_model(self):
        self.policy.del_model()

    def is_saved_model(self) -> bool:
        return self.policy.is_saved_model()

    def get_results_path(self):
        return self.policy.result_path
