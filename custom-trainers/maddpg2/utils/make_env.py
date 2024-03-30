from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.environment import UnityEnvironment
import numpy as np


class PZ_Wrapper(UnityParallelEnv):
    """ Wrapper for UnityParallelEnv to make it compatible with the implementation
        of MADDPG.
        It mostly converts in / outputs for the environment to be compatible with
        the MADDPG implementation.
    """

    def reset_env(self, agent_ids):
        """ Reset the environment and return the initial observations.
            Convert the observations to a format compatible with MADDPG.

        Args:
            agent_ids (List[str]): List of agent_ids from MADDPG instance
        """
        # giving agent_ids to make sure we keep the order of agents
        temp = UnityParallelEnv.reset(self)  # Dict agent_id -> 1D array
        return PZ_Wrapper.convert_obs(temp, agent_ids)

    def step_env(self, actions, agent_ids):
        """ Step the environment with the given actions and return the next
            observations, rewards, dones and infos.
            Convert the actions to a format compatible with the environment.
            Convert the observations to a format compatible with MADDPG.

        Args:
            actions (_type_): Actions given by MADDPG instance
            agent_ids (List[str]): List of agent_ids from MADDPG instance

        Returns:
            Tuple: next_obs, rewards, dones, infos
        """
        # giving agent_ids to make sure we keep the order of agents
        to_step = {agent: actions[i].flatten().astype(np.float64)
                   for i, agent in enumerate(agent_ids)}
        tmp_next_obs, tmp_rewards, tmp_dones, infos = UnityParallelEnv.step(self, to_step)
        next_obs = PZ_Wrapper.convert_obs(tmp_next_obs, agent_ids)
        rewards = PZ_Wrapper.convert_obs(tmp_rewards, agent_ids)
        dones = PZ_Wrapper.convert_obs(tmp_dones, agent_ids)
        return next_obs, rewards, dones, infos

    @staticmethod
    def convert_obs(obs, agent_ids):
        """ Convert the env observations to a format compatible with MADDPG.
        """
        return np.array([[obs[agent_id] for agent_id in agent_ids]])


def make_env(executable=None, seed=None, benchmark=False, discrete_action=False):
    '''
    Creates a Wrapped Unity Parallel environment for PettingZoo.

    Input:
        executable      :   the path of the unity exectuable. If none, wait for
                            the user to press play in the unity editor
        seed            :   the random seed for the environment
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
        discrete_action :   whether the actions should be discrete.
    Output:
        pz_env          :  a Unity Wrapped PettingZoo Parallel environment
    '''
    u_env = UnityEnvironment(file_name=executable, seed=seed)
    pz_env = PZ_Wrapper(u_env)
    print("Environnment loaded : \n")
    print("\tAgent names:", pz_env.agents)
    print("\tFirst agent:", pz_env.agents[0])
    print("\tObservation space of first agent:", pz_env.observation_spaces[pz_env.agents[0]].shape)
    print("\tAction space of first agent:", pz_env.action_spaces[pz_env.agents[0]])
    pz_env.reset_env(pz_env.agents)

    return pz_env
