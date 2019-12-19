from gym.envs.registration import register

register(
    id='Discrete-1-Equity-v0',
    entry_point='drltr.envs:Discrete1Equity',
)

register(
    id='Discrete-1-Equity-Short-v0',
    entry_point='drltr.envs:Discrete1EquityShort',
)

register(
    id='Discrete-1-Equity-Costs-v0',
    entry_point='drltr.envs:Discrete1EquityCosts',
)

register(
    id='Discrete-2-Equities-v0',
    entry_point='drltr.envs:Discrete2Equities',
)
