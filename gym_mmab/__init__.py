from gym.envs.registration import register

register(
    id='mmab-v0',
    entry_point='gym_mmab.envs:MMAB',
)
