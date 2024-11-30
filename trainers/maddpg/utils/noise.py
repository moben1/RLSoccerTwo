""" Noise module for exploration.
"""
import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """ Ornstein-Uhlenbeck noise process. Used for exploration.
    """

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.05):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        """ Reset the internal state (= noise) to mean (mu).
        """
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """ Update internal state and return it as a noise sample.
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
