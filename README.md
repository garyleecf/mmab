# Multiplayer Multi-Armed bandit (Deterministic Reward)

Instructions:
```bash
git clone https://github.com/garyleecf/mmab.git
cd mmab
pip install -e .
```

In Python:
```python
import gym
env = gym.make('gym_mmab:mmab-v0', n_players=3, n_arms=5) # ; or
env = gym.make('gym_cardgame:mmab-v0') # If you leave arguments empty, default: n_players=3, n_arms=10

# Game Simulation Starts:
observations = env.reset()
env.render()

# For actual game play, iterate for n_steps:
actions = np.zeros(env.n_players, dtype=int)
for n in range(env.n_players):
    actions[n] = agent_list[n].choose_action()
observation, reward, done, info = env.step(actions)


```
See testrun_mmab.py for how the environment and agents are used.
