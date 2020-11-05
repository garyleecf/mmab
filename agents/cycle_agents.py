import random
import numpy as np
from agents import Agent


# "CycleAgent" extends the base Agent. This will be the players you will "play with".
# Try to see if you can make sense of what's happening here:
class CycleAgent(Agent):
    def __init__(self, id, n_arms, offset=None):
        super().__init__(id, n_arms)
        self.reset(offset)
        self.clock = 0

    def reset(self, offset=None):
        super().reset()
        if offset is None:
            self.clock_offset = np.random.randint(self.n_arms)
        else:
            self.clock_offset = offset
        self.clock = 0

    def observe(self, observation):
        pass

    def choose_action(self):
        self.clock += 1
        action = (self.clock + self.clock_offset) % self.n_arms
        self.last_action = action
        return action
