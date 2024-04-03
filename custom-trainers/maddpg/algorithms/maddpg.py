""" MADDPG algorithm implementation
"""
import logging
import torch
from gym.spaces import Box

from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()


class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """

    def __init__(self, agent_init_params, gamma=0.95, tau=0.01,
                 actor_lr=5e-4, critic_lr=1e-3,
                 hidden_units=64, discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(agent_init_params)
        self.agent_ids = [param['agent_id'] for param in agent_init_params]
        self.agents = [DDPGAgent(actor_lr=actor_lr, critic_lr=critic_lr,
                                 discrete_action=discrete_action,
                                 hidden_units=hidden_units, **params)
                       for params in agent_init_params]

        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.discrete_action = discrete_action

        # default devices are not important since it is modified
        # at each prep_training() and prep_rollouts() call
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        """ Return a list of the policies of each ddpg agent
        """
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        """ Return a list of the target policies of each ddpg agent
        """
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        """ Reset noise for each agent
        """
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                               observations)]

    def update(self, sample, agent_i, parallel=False):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.discrete_action:  # one-hot encode action
            all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                            zip(self.target_policies, next_obs)]
        else:
            all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                         next_obs)]
        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma
                        * curr_agent.target_critic(trgt_vf_in)
                        * (1 - dones[agent_i].view(-1, 1)))

        vf_in = torch.cat((*obs, *acs), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out

        all_pol_acs = []
        for i, pi, ob in zip(range(self.nagents), self.policies, obs):
            if i == agent_i:
                all_pol_acs.append(curr_pol_vf_in)
            elif self.discrete_action:
                all_pol_acs.append(onehot_from_logits(pi(ob)))
            else:
                all_pol_acs.append(pi(ob))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        stats = {'vf_loss': vf_loss,
                 'pol_loss': pol_loss,
                 'it': self.niter}
        logging.debug("Updating %s", stats)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device: str = 'cuda') -> None:
        """ Prepare networks for training, load to device if necessary

        Args:
            device (str): Device to load the networks to for training. 
                          Defaults to 'cuda'.
        """
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        #if device == 'gpu':
        #    def fn(x): return x.cuda()
        #else:
        #    def fn(x): return x.cpu()
        torch_device = torch.device(device)
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy.to(torch_device)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic.to(torch_device)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy.to(torch_device)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic.to(torch_device)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device: str = 'cpu') -> None:
        """ Prepare networks for rollouts (inference without training), 
            load to device if necessary

        Args:
            device (str): Device to load the networks to for rollout. 
                          Defaults to 'cpu'.

        Returns:
            _type_: _description_
        """
        for a in self.agents:
            a.policy.eval()
        #if device == 'gpu':
        #    def fn(x): return x.cuda()
        #else:
        #    def fn(x): return x.cpu()
        ## only need main policy for rollouts
        if device == 'cuda':
            torch_device = torch.device('cuda')
        else:
            torch_device = torch.device('cpu')
        if not self.pol_dev == device:
            for a in self.agents:
                #a.policy = fn(a.policy)
                a.policy.to(torch_device)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01, actor_lr=5e-4,
                      critic_lr=1e-3, hidden_units=128):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        for a_id, acsp, obsp in zip(env.agent_ids, env.action_space, env.observation_space):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                def get_shape(x): return x.shape[0]
            else:  # Discrete
                discrete_action = True
                def get_shape(x): return x.n
            num_out_pol = get_shape(acsp)
            num_in_critic = 0
            for oobsp in env.observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env.action_space:
                num_in_critic += get_shape(oacsp)
            agent_init_params.append({'agent_id': a_id,
                                      'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})

        init_dict = {'gamma': gamma, 'tau': tau,
                     'actor_lr': actor_lr, 'critic_lr': critic_lr,
                     'hidden_units': hidden_units,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance
