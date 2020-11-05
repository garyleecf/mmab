import random
import numpy as np
from agents import Agent


# "RandomAgent" extends the base Agent.
class RandomAgent(Agent):
    def __init__(self, id, n_arms):
        super().__init__(id, n_arms)

    def reset(self, offset=None):
        super().reset()

    def observe(self, observation):
        pass

    def choose_action(self):
        return np.random.randint(self.n_arms)
