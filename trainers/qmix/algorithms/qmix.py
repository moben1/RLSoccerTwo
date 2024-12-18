import os

import torch
from torch import Tensor

from utils.networks import RNN, QMixNet, weight_init


train_config = {
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
    "cuda" : False,
}


class QMix(object):

    def __init__(self, env_info):
        self.n_agents = len(env_info.agents)
        self.n_actions = env_info.action_spaces[env_info.agents[0]].shape[0]
        state_space = env_info.observation_spaces[env_info.agents[0]].shape[0]
        input_shape = state_space + self.n_agents + self.n_actions

        # Neural network configuration
        self.rnn_hidden_dim = 64
        # The network agent for each agent to select the action
        # TODO: adjust output of rnn for multidiscrete
        self.rnn_eval = RNN(input_shape, self.n_actions, self.rnn_hidden_dim)
        self.rnn_target = RNN(input_shape, self.n_actions, self.rnn_hidden_dim)
        # The network for mixing the q value of agents
        self.qmix_net_eval = QMixNet(self.n_agents, state_space)
        self.qmix_net_target = QMixNet(self.n_agents, state_space)
        self.init_weight()
        self.eval_parameters = list(self.qmix_net_eval.parameters()) + list(self.rnn_eval.parameters())
        self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=train_config['lr_critic'])

        # Initialize the path to save the model and the result
        self.model_path = os.path.join(train_config['model_dir'], "qmix")
        self.result_path = os.path.join(train_config['result_dir'], "qmix")
        self.rnn_eval_path = os.path.join(self.model_path, "rnn_eval.pth")
        self.rnn_target_path = os.path.join(self.model_path, "rnn_target.pth")
        self.qmix_net_eval_path = os.path.join(self.model_path, "qmix_net_eval.pth")
        self.qmix_net_target_path = os.path.join(self.model_path, "qmix_net_target.pth")

        # GPU configuration
        if train_config['cuda']:
            torch.cuda.empty_cache()
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        self.rnn_eval.to(self.device)
        self.rnn_target.to(self.device)
        self.qmix_net_eval.to(self.device)
        self.qmix_net_target.to(self.device)

    def init_weight(self):
        self.rnn_eval.apply(weight_init)
        self.rnn_target.apply(weight_init)
        self.qmix_net_eval.apply(weight_init)
        self.qmix_net_target.apply(weight_init)

    def learn(self, batch_data: dict, episode_num: int):
        obs = batch_data['obs'].to(self.device)
        obs_next = batch_data['obs_next'].to(self.device)
        state = batch_data['state'].to(self.device)
        state_next = batch_data['state_next'].to(self.device)
        rewards = batch_data['rewards'].unsqueeze(dim=-1).to(self.device)
        actions = batch_data['actions'].long().to(self.device)
        actions_onehot = batch_data['actions_onehot'].to(self.device)
        terminated = batch_data['terminated'].unsqueeze(dim=-1).to(self.device)

        q_evals, q_targets = [], []
        batch_size = batch_data['sample_size']
        self.init_hidden(batch_size)
        for i in range(batch_data['max_step']):
            inputs, inputs_next = self._get_inputs(batch_size, i, obs[:, i], obs_next[:, i],
                                                   actions_onehot)
            q_eval, self.eval_hidden = self.rnn_eval(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.rnn_target(inputs_next, self.target_hidden)
            # reshape the Q values to separate by `n_agents`
            q_eval = q_eval.view(batch_size, self.n_agents, -1)
            q_target = q_target.view(batch_size, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # Aggregate the Q values obtained above
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        # Find the Q values corresponding to the selected actions
        q_evals = torch.gather(q_evals, dim=3, index=actions).squeeze(3)
        q_targets = q_targets.max(dim=3)[0]
        # Input the Q values and state into the mix network
        q_total_eval = self.qmix_net_eval(q_evals, state)
        q_total_target = self.qmix_net_target(q_targets, state_next)

        targets = rewards + train_config['gamma'] * q_total_target * terminated
        td_error = (q_total_eval - targets.detach())
        # Erase the TD error of the padded experiences
        masked_td_error = terminated * td_error
        # calculate the loss function
        loss = (masked_td_error ** 2).sum() / terminated.sum()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, train_config['grad_norm_clip'])
        self.optimizer.step()
        if episode_num > 0 and episode_num % train_config['target_update_cycle'] == 0:
            self.rnn_target.load_state_dict(self.rnn_eval.state_dict())
            self.qmix_net_target.load_state_dict(self.qmix_net_eval.state_dict())

    def _get_inputs(self, batch_size: int, batch_index: int, obs: Tensor, obs_next: Tensor,
                    actions_onehot: Tensor) -> tuple:
        """
            get the input values for the Q network, incorporate the actions into the observations
            :return:
        """
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        if batch_index == 0:
            inputs.append(torch.zeros_like(actions_onehot[:, batch_index]))
        else:
            inputs.append(actions_onehot[:, batch_index - 1])
        inputs_next.append(actions_onehot[:, batch_index])
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device))
        inputs = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(batch_size * self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def init_hidden(self, batch_size):
        # initialize an `eval_hidden` and `target_hidden` for each agent in each episode
        self.eval_hidden = torch.zeros((batch_size, self.n_agents, self.rnn_hidden_dim)).to(self.device)
        self.target_hidden = torch.zeros((batch_size, self.n_agents, self.rnn_hidden_dim)).to(self.device)

    def save_model(self):
        torch.save(self.rnn_eval.state_dict(), self.rnn_eval_path)
        torch.save(self.rnn_target.state_dict(), self.rnn_target_path)
        torch.save(self.qmix_net_eval.state_dict(), self.qmix_net_eval_path)
        torch.save(self.qmix_net_target.state_dict(), self.qmix_net_target_path)

    def load_model(self):
        self.rnn_eval.load_state_dict(torch.load(self.rnn_eval_path))
        self.rnn_target.load_state_dict(torch.load(self.rnn_target_path))
        self.qmix_net_eval.load_state_dict(torch.load(self.qmix_net_eval_path))
        self.qmix_net_target.load_state_dict(torch.load(self.qmix_net_target_path))

    def del_model(self):
        file_list = os.listdir(self.model_path)
        for file in file_list:
            os.remove(os.path.join(self.model_path, file))

    def is_saved_model(self) -> bool:
        return os.path.exists(self.rnn_eval_path) and os.path.exists(
            self.rnn_target_path) and os.path.exists(self.qmix_net_eval_path) and os.path.exists(
            self.qmix_net_target_path)
