""" 
Functions to instanciate single or parallel environments for Unity
"""
from typing import Union
import numpy as np

from mlagents_envs.environment import UnityEnvironment

from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from utils.PZWrapper import PZWrapper


def make_env(executable: str, seed: int, worker: int = 0, no_graphics: bool = False) -> PZWrapper:
    """_summary_

    Args:
        executable (str): Path to the Unity executable, if None, use play in editor.
        seed (seed): Random seed for the environment
        worker (int): Unique worker id for the environment
        no_graphics (bool): Whether to run the environment with graphics or not

    Returns:
        PZWrapper: Specilised PettingZoo wrapper for the Unity environment
    """
    u_env = UnityEnvironment(file_name=executable, seed=seed,
                             worker_id=worker, no_graphics=no_graphics)
    pz_env = PZWrapper(u_env)
    pz_env.reset_env(pz_env.agents)

    return pz_env


def make_parallel_env(executable: str, n_rollout_threads: int, use_subprocess: bool,
                      seed: int, no_graphics: bool) -> Union[SubprocVecEnv, DummyVecEnv]:
    """ Create parallel instances of a Unity environment wrapped with PettingZoo.

    Args:
        executable (str): Path to the Unity executable
        n_rollout_threads (int): Number of parallel instances
        use_subprocess (bool): Whether to use multiprocessing or not
        seed (int): initial random seed for the environment
        no_graphics (bool): Whether to run the environment with graphics or not

    Returns:
        Union[SubprocVecEnv, DummyVecEnv]: Parallel environments
    """
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
