""" Specialisation of UnityParallelEnv to make it compatible with MADDPG.
"""
from typing import Tuple
import numpy as np

from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.envs.env_helpers import _agent_id_to_behavior

from gym import error, spaces


class PZWrapper(UnityParallelEnv):
    """ Wrapper for UnityParallelEnv to make it compatible with the implementation
        of MADDPG.
        It mostly converts in / outputs for the environment to be compatible with
        the MADDPG implementation.
        Also fix some unexpected behavior from UnityParallelEnv.
    """

    def reset_env(self, agent_ids: list):
        """ Reset the environment and return the initial observations.
            Convert the observations to a format compatible with MADDPG.
            Sending agent_ids to make sure we keep the order of agents.

        Args:
            agent_ids (List[str]): List of agent_ids from MADDPG instance

        Returns:
            np.array: Initial observations
        """
        # giving agent_ids to make sure we keep the order of agents
        temp = UnityParallelEnv.reset(self)
        return PZWrapper.convert_obs(temp, agent_ids)

    def step_env(self, actions: list, agent_ids: list):
        """ Step the environment with the given actions and return the next
            observations, rewards, dones and infos.
            Convert the actions to a format compatible with the environment.
            Convert the observations to a format compatible with MADDPG.
            Sending agent_ids to make sure we keep the order of agents.

        Args:
            actions (List[str]): Actions given by MADDPG instance
            agent_ids (List[str]): List of agent_ids from MADDPG instance

        Returns:
            Tuple: next_obs, rewards, dones, infos
        """
        # giving agent_ids to make sure we keep the order of agents
        to_step = {agent: actions[i].flatten()
                   for i, agent in enumerate(agent_ids)}
        tmp_next_obs, tmp_rewards, tmp_dones, infos = UnityParallelEnv.step(self, to_step)
        next_obs = PZWrapper.convert_obs(tmp_next_obs, agent_ids)
        rewards = PZWrapper.convert_obs(tmp_rewards, agent_ids)
        dones = PZWrapper.convert_obs(tmp_dones, agent_ids)
        return next_obs, rewards, dones, infos

    def _update_action_spaces(self) -> None:
        """ Redefining to initialize action space as float32
        """
        self._assert_loaded()
        for behavior_name in self._env.behavior_specs.keys():
            if behavior_name not in self._action_spaces:
                act_spec = self._env.behavior_specs[behavior_name].action_spec
                if (
                    act_spec.continuous_size == 0
                    and len(act_spec.discrete_branches) == 0
                ):
                    raise error.Error("No actions found")
                if act_spec.discrete_size == 1:
                    d_space = spaces.Discrete(act_spec.discrete_branches[0])
                    if self._seed is not None:
                        d_space.seed(self._seed)
                    if act_spec.continuous_size == 0:
                        self._action_spaces[behavior_name] = d_space
                        continue
                if act_spec.discrete_size > 0:
                    d_space = spaces.MultiDiscrete(act_spec.discrete_branches)
                    if self._seed is not None:
                        d_space.seed(self._seed)
                    if act_spec.continuous_size == 0:
                        self._action_spaces[behavior_name] = d_space
                        continue
                if act_spec.continuous_size > 0:
                    c_space = spaces.Box(
                        -1, 1, (act_spec.continuous_size,), dtype=np.float32
                    )
                    if self._seed is not None:
                        c_space.seed(self._seed)
                    if len(act_spec.discrete_branches) == 0:
                        self._action_spaces[behavior_name] = c_space
                        continue
                self._action_spaces[behavior_name] = spaces.Tuple((c_space, d_space))

    def _process_action(self, current_agent, action):
        """ Redefining cause original function send the same value for each
            action entry in the tuple.
        """
        current_action_space = self.action_space(current_agent)
        # Convert actions
        if action is not None:
            if isinstance(action, Tuple):
                action = tuple(np.array(a) for a in action)
            else:
                action = self._action_to_np(current_action_space, action)
            if not current_action_space.contains(action):  # type: ignore
                raise error.Error(f"Invalid action, got {action} but "
                                  f"was expecting action from {self.action_space}")
            if isinstance(current_action_space, spaces.Tuple):
                action = ActionTuple(action[0], action[1])
            elif isinstance(current_action_space, spaces.MultiDiscrete):
                action = ActionTuple(None, action)
            elif isinstance(current_action_space, spaces.Discrete):
                action = ActionTuple(None, np.array(action).reshape(1, 1))
            else:
                action = ActionTuple(action, None)

        if not self._dones[current_agent]:
            current_behavior = _agent_id_to_behavior(current_agent)
            current_index = self._agent_id_to_index[current_agent]
            if action.continuous is not None:
                self._current_action[current_behavior].continuous[
                    current_index
                ] = action.continuous
            if action.discrete is not None:
                self._current_action[current_behavior].discrete[
                    current_index
                ] = action.discrete[0]
        else:
            self._live_agents.remove(current_agent)
            del self._observations[current_agent]
            del self._dones[current_agent]
            del self._rewards[current_agent]
            del self._cumm_rewards[current_agent]
            del self._infos[current_agent]

    @staticmethod
    def convert_obs(obs, agent_ids: list):
        """ Convert the env observations to a format compatible with MADDPG.
        """
        return np.array([obs[agent_id] for agent_id in agent_ids])
