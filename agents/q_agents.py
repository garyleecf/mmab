import random
import numpy as np
from agents import Agent

# In what follows, I have implemented a "smarter" agent that actually learns from past actions and observations.
# This uses concepts related to "Q Learning" -- we will touch on this in subsequent weeks!
# However, if you're curious, read on, try and figure out what's happening and how this QAgent actually learns ("observe").
class QAgent(Agent):
    def __init__(self, id, n_arms, offset=None):
        super().__init__(id, n_arms)
        self.reset()

    def reset(self):
        super().reset()
        self.clock = 0
        self.curr_actions = None
        self.q_table = np.zeros((self.n_arms, self.n_arms, self.n_arms)) # specifically for 3player game
        self.epsilon = 0.8
        self.gamma = 0.0
        self.alpha = 0.5 # learning rate

    def observe(self, observation):
        obs_rewards = observation["Rewards"]
        collisions = observation["Collisions"]
        actions = observation["Actions"]
        prev_actions = observation["PreviousActions"]
        if prev_actions is not None:
            my_reward = obs_rewards[self.id]
            my_reward -= collisions[self.id] # "reward shaping", avoiding collisions by penalizing it
            old_q_value = self.q_table[actions[0], prev_actions[1], prev_actions[2]]
            est_opt_value = np.max(self.q_table[:, actions[1], actions[2]])
            self.q_table[actions[0], prev_actions[1], prev_actions[2]] += self.alpha * (my_reward + self.gamma*est_opt_value - old_q_value)

        self.curr_actions = np.copy(actions)

        if (self.clock+1)%100 == 0:
            new_epsilon = self.epsilon/2
            self.epsilon = max(new_epsilon, 0.0001)

    def choose_action(self):
        self.clock += 1
        action = np.random.randint(self.n_arms)
        if np.random.rand() < 1-self.epsilon and self.curr_actions is not None: # epsilon greedy
            action = np.argmax(self.q_table[:, self.curr_actions[1], self.curr_actions[2]])
        self.last_action = action
        return action
