""" Interfaces dor managing multiple instances of pettingzoo environments
"""
from multiprocessing import Process, Pipe
import numpy as np

from external.vec_env import VecEnv, CloudpickleWrapper


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if all(done):
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send(([env.observation_space(a) for a in env.agent_ids],
                        [env.action_space(a) for a in env.agent_ids]))
        elif cmd == 'get_agent_ids':
            remote.send(env.agent_ids)
        elif cmd == 'scale_float_property':
            env.scale_float_property(*data)
        elif cmd == 'set_time_scale':
            env.set_time_scale(data)
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    """ VecEnv that runs multiple environments in parallel in subproceses 
        and communicates with them via pipes
    """

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of pettingzoo environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn)
                   in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_agent_ids', None))
        agent_ids = self.remotes[0].recv()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        VecEnv.__init__(self, len(env_fns), observation_space, action_space, agent_ids)

    def scale_float_property(self, property_name: str, scale: float):
        """ Scale the float property of each environment.

        Args:
            property_name (str): Name of the property to scale
            scale (float): Scale factor
        """
        for remote in self.remotes:
            remote.send(('scale_float_property', (property_name, scale)))

    def set_time_scale(self, scale: float):
        """ Set the time scale of each environment.

        Args:
            scale (float): Time scale factor
        """
        for remote in self.remotes:
            remote.send(('set_time_scale', scale))

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv):
    """ Run multiple environments in parallel, but sequentially
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns),
                        [env.observation_space(a) for a in env.agent_ids],
                        [env.action_space(a) for a in env.agent_ids],
                        env.agent_ids)
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.actions = None

    def scale_float_property(self, property_name: str, scale: float):
        """ Scale the float property of each environment.

        Args:
            property_name (str): Name of the property to scale
            scale (float): Scale factor
        """
        for env in self.envs:
            env.scale_float_property(property_name, scale)

    def set_time_scale(self, scale: float):
        """ Set the time scale of each environment.

        Args:
            scale (float): Time scale factor
        """
        for env in self.envs:
            env.set_time_scale(scale)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))
        self.ts += 1
        for (i, done) in enumerate(dones):
            if all(done):
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        self.actions = None
        return np.array(obs), np.array(rews), np.array(dones), infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.array(results)

    def close(self):
        return
