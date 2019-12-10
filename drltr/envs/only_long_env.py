import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 500
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000

LOOKBACK_NUM = 5


class OnlyLongEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, ep_len):
        super(OnlyLongEnv, self).__init__()

        self.df = df
        self.ep_len = ep_len

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Discrete(4)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, LOOKBACK_NUM + 1), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step - LOOKBACK_NUM:self.current_step, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOKBACK_NUM:self.current_step, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOKBACK_NUM:self.current_step, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOKBACK_NUM:self.current_step, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - LOOKBACK_NUM:self.current_step, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # print(frame)
        # print(self.current_step)

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs[:, :, np.newaxis]

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
        self.last_price = current_price

        new_position = action

        position_change = new_position - self.shares_held
        prev_cost = self.cost_basis * self.shares_held
        new_cost = position_change * current_price

        self.balance -= new_cost
        if new_position == 0:
            self.cost_basis = 0
        else:
            self.cost_basis = (prev_cost + new_cost) / new_position

        self.shares_held = new_position

        if position_change < 0:
            self.total_shares_sold += abs(position_change)
            self.total_sales_value += abs(position_change * current_price)

        self.net_worth = self.balance + self.shares_held * current_price
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        initial_worth = self.net_worth

        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.t += 1

        # delay_modifier = (self.current_step / MAX_STEPS)
        #
        # reward = self.balance * delay_modifier
        reward = (self.net_worth - initial_worth) / self.last_price
        done = self.net_worth <= 0 or self.t == self.ep_len

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            LOOKBACK_NUM, len(self.df.loc[:, 'Open'].values) - self.ep_len)
        self.t = 0

        self.last_price = random.uniform(
            self.df.loc[self.current_step - 1, "Open"], self.df.loc[self.current_step - 1, "Close"])

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print('Step: {}'.format(self.current_step))
        print('Balance: {}'.format(self.balance))
        print(
            'Shares held: {} (Total sold: {})'.format(self.shares_held,
                                                      self.total_shares_sold))
        print(
            'Avg cost for held shares: {} (Total sales value: {})'.format(self.cost_basis,
                                                                          self.total_sales_value))
        print(
            'Net worth: {} (Max net worth: {})'.format(self.net_worth,
                                                       self.max_net_worth))
        print('Profit: {}'.format(profit))
