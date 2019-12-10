from gym.envs.registration import register

register(
    id='OnlyLong-v0',
    entry_point='drltr.envs:OnlyLongEnv',
)
