""" 
Functions to instanciate single or parallel environments for Unity
"""
import numpy as np

from mlagents_envs.environment import UnityEnvironment

from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.PZWrapper import PZWrapper


def make_env(executable=None, seed=None, worker=0, benchmark=False, no_graphics=False):
    '''
    Creates a Wrapped Unity Parallel environment for PettingZoo.

    Input:
        executable      :   the path of the unity exectuable. If none, wait for
                            the user to press play in the unity editor
        seed            :   the random seed for the environment
        worker          :   unity worker to connect with
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)
        discrete_action :   whether the actions should be discrete.
    Output:
        pz_env          :  a Unity Wrapped PettingZoo Parallel environment
    '''
    u_env = UnityEnvironment(file_name=executable, seed=seed, worker_id=worker, no_graphics=no_graphics)
    pz_env = PZWrapper(u_env)
    pz_env.reset_env(pz_env.agents)

    return pz_env


def make_parallel_env(executable, n_rollout_threads, use_subprocess, seed, no_graphics):
    def get_env_fn(rank):
        def init_env():
            s = seed + rank * 1000
            env = make_env(executable, seed=s, worker=rank, no_graphics=no_graphics)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if not use_subprocess:
        return DummyVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
    return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])
