import gym
import agents
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) # this setting helps me print floats in numpy array only up to 3 d.p.

_DEBUG = False
n_steps = 1000

def main():
    env = gym.make('gym_mmab:mmab-v0', n_players=3, n_arms=5)
    # env = gym.make('gym_cardgame:mmab-v0') # If you leave arguments empty, default: n_players=3, n_arms=10


    agent0_candidates = [agents.Agent(0, env.n_arms), agents.RandomAgent(0, env.n_arms), agents.QAgent(0, env.n_arms)]
    saved_obs_history = []
    for agent0 in agent0_candidates:
        print("====================")
        print(f"Testing with {type(agent0).__name__}")
        print("====================")
        obs_history = test_selectedagent(env, agent0)
        saved_obs_history.append(obs_history)
        print()

    print("====================")
    print("Comparing cumulative rewards in last 100 rounds")
    for n, agent0 in enumerate(agent0_candidates):
        tot_rewards = np.array(saved_obs_history[n][-100:]).sum(axis=0)[0]
        print(f"{type(agent0).__name__:>14}: {tot_rewards}")
    print()
    
def test_selectedagent(env, agent0):
    env.reset()

    agent_list = [agent0]
    for id in range(1, env.n_players):
        new_agent = agents.CycleAgent(id, env.n_arms, offset=id)
        # new_agent = RandomAgent(id, n_arms)
        agent_list.append(new_agent)


    prev_actions = None
    for t in range(n_steps):
        actions = np.zeros(env.n_players, dtype=int)
        for n in range(env.n_players):
            actions[n] = agent_list[n].choose_action()
        observation, reward, done, info = env.step(actions)

        # For simplicity, let's also assume that each agent
        # (i.e., our learning agent) can observe what actions
        # the others had taken this turn (after choosing our own action, of course)
        # That is, I want to learn how to decide my next action based on other players' action history
        observation["Actions"] = actions
        observation["PreviousActions"] = prev_actions
        agent0.observe(observation)
        prev_actions = actions
        if t >= n_steps - 10:
            env.render()

    return env.obs_history

if __name__ == '__main__':
    main()
