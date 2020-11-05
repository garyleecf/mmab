import gym
from gym import spaces
import random
import math
import numpy as np
from collections import deque

class MMAB(gym.Env):
    """Multi Agent Multi-Armed Bandit -- Custom Environment that follows gym interface"""

    def __init__(self, n_players=3, n_arms=10):
        super().__init__() # This runs the __init__ lines for the parent class, i.e. gym.Env

        self.n_players = n_players
        self.n_arms = n_arms

        # Refer to recommended readings on defining action_space and observation_space;
        # We will eventually find it useful, and it's good to implement it
        # However, as we start off, you can also omit this first, and we'll revisit again later
        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_players,))

        # When we initialize the game, we randomly generate what those rewards are.
        # This will remain constant throughout the course of a single game, across multiple turns
        # Remember that these rewards are generated once at the start of the game, but are not known to the players at the start
        self.reward_per_arm = [10*np.random.rand() for _ in range(self.n_arms)]

        # Instead of repeating the lines of code to reset the game, why not just put them all under "reset", and call it here!
        self.reset()

    def reset(self):
        self.clock = 0

        self.action_history = []
        self.obs_history = []
        self.collision_history = []

        collisions = np.empty(self.n_players)*np.nan
        obs_rewards = np.empty(self.n_players)*np.nan
        observation = {"Rewards": obs_rewards, "Collisions":collisions}
        return observation

    def step(self, actions):
        self.clock += 1
        self._check_valid_actions(actions)

        collisions = self._check_collisions(actions)
        arm_rewards = np.array([self.reward_per_arm[a] for a in actions])
        obs_rewards = (1-collisions)*arm_rewards

        self.action_history.append(actions)
        self.obs_history.append(obs_rewards)
        self.collision_history.append(collisions)

        observation = {"Rewards": obs_rewards, "Collisions":collisions}
        info = {"Rewards": obs_rewards, "Collisions":collisions, "ArmReward":self.reward_per_arm, "Clock":self.clock}
        reward = obs_rewards
        done = False
        return observation, reward, done, info

    # I created 2 helper functions that does the steps as the name implies
    # -- checking if the action is valid, and checking if there's "collisions" (two players choosing the same arm)
    # It's always good practice to modularize code; also, if you are stuck, this also helps with pseudo-coding
    # (e.g. under step, I know that I want to check for collisions, so I can create a function for it.
    # However, I don't know specifically how to do it yet, and so I will leave the function blank, and seek help on how to do this step!)
    def _check_valid_actions(self, actions):
        for i, a in enumerate(actions):
            assert self.action_space.contains(a), f'Invalid action by Player{i}: {a}'

    def _check_collisions(self, actions):
        unique, counts = np.unique(actions, return_counts=True)
        freq_lookup = {k:v for k,v in zip(unique, counts)}
        return np.array([freq_lookup[a]>1 for a in actions]) # True if the i-th player's action is also chosen by some other players (in actions)

    # render is actually another critical function in a gym environment; this is typically called to "show" the state of the game/environment.
    # In more complex RL environments, you will see graphical libraries used to render the state of the game.
    # In this case, I'm going with a simple print out on what happened in this turn.
    def render(self, window=10):
        print(f"Iter {self.clock:>5}: Action {self.action_history[-1]} Reward {self.obs_history[-1]}  Collisions {self.collision_history[-1]}")
