import random

# Base Agent class:
class Agent():
    # General design I use: __init__, reset,
    # choose_action (called before env.step), observe (called after env.step and receiving the observations)
    def __init__(self, id, n_arms):
        self.id = id
        self.n_arms = n_arms
        self.reset()

    def reset(self):
        self.last_action = None

    def observe(self, observation):
        obs_rewards = observation["Rewards"]
        collisions = observation["Collisions"]
        pass

    def choose_action(self):
        self.last_action = 0
        return 0
