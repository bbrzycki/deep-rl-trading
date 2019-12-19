import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

HIGH_LIM = 2147483647
MAX_ACCOUNT_BALANCE = 100000000
MAX_NUM_SHARES = 100000000
MAX_SHARE_PRICE = 1000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class Discrete2Equities(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Discrete2Equities, self).__init__()

        # self.df = df
        # self.ep_len = ep_len

        self.lookback_num = 5

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        self.action_space = spaces.Discrete(16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, (self.lookback_num + 1) * 2 + 1), dtype=np.float16)

    def set_init(self, df1, df2, ep_len, lookback_num):
        self.df1 = df1
        self.df2 = df2
        self.ep_len = ep_len
        self.lookback_num = lookback_num
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, (lookback_num + 1) * 2 + 1), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df1.loc[self.current_step - self.lookback_num:self.current_step, 'Open'].values / MAX_SHARE_PRICE,
            self.df1.loc[self.current_step - self.lookback_num:self.current_step, 'High'].values / MAX_SHARE_PRICE,
            self.df1.loc[self.current_step - self.lookback_num:self.current_step, 'Low'].values / MAX_SHARE_PRICE,
            self.df1.loc[self.current_step - self.lookback_num:self.current_step, 'Close'].values / MAX_SHARE_PRICE,
            self.df1.loc[self.current_step - self.lookback_num:self.current_step, 'Volume'].values / MAX_NUM_SHARES,
        ])

        combined_frame = np.append(frame, np.array([
            self.df2.loc[self.current_step - self.lookback_num:self.current_step, 'Open'].values / MAX_SHARE_PRICE,
            self.df2.loc[self.current_step - self.lookback_num:self.current_step, 'High'].values / MAX_SHARE_PRICE,
            self.df2.loc[self.current_step - self.lookback_num:self.current_step, 'Low'].values / MAX_SHARE_PRICE,
            self.df2.loc[self.current_step - self.lookback_num:self.current_step, 'Close'].values / MAX_SHARE_PRICE,
            self.df2.loc[self.current_step - self.lookback_num:self.current_step, 'Volume'].values / MAX_NUM_SHARES,
        ]), axis=1)

        # print(frame)
        # print(self.current_step)

        # Append additional data and scale each value to between 0-1
        obs = np.append(combined_frame, np.array([[
            self.net_worth / MAX_ACCOUNT_BALANCE,
            # self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held1 / 3,
            self.shares_held2 / 3,
            self.cost_basis1 / MAX_SHARE_PRICE,
            self.cost_basis2 / MAX_SHARE_PRICE,
            # self.total_shares_sold / MAX_NUM_SHARES,
            # self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
            # (self.price_change + 1) / 100,
        ]]).transpose(), axis=1)

        return obs[:, :, np.newaxis]

    def _take_action(self, action):
        self.last_price1 = self.current_price1
        self.last_price2 = self.current_price2

        # Set the current price to a random price within the time step
        self.current_price1 = random.uniform(
            self.df1.loc[self.current_step, "High"], self.df1.loc[self.current_step, "Low"])
        # self.current_price1 = self.df1.loc[self.current_step, "Close"]

        self.current_price2 = random.uniform(
            self.df2.loc[self.current_step, "High"], self.df2.loc[self.current_step, "Low"])
        # self.current_price2 = self.df2.loc[self.current_step, "Close"]

        self.price_change1 = (self.current_price1 - self.last_price1) / self.last_price1
        self.price_change2 = (self.current_price2 - self.last_price2) / self.last_price2

        # Positions can take on values from 0 to 3
        new_position1 = action // 4
        new_position2 = action % 4

        self.position_change1 = new_position1 - self.shares_held1
        self.position_change2 = new_position2 - self.shares_held2

        prev_cost1 = self.cost_basis1 * self.shares_held1
        new_cost1 = self.position_change1 * self.current_price1

        prev_cost2 = self.cost_basis2 * self.shares_held2
        new_cost2 = self.position_change2 * self.current_price2

        self.balance -= new_cost1 + new_cost2
        if new_position1 == 0:
            self.cost_basis1 = 0
        else:
            self.cost_basis1 = (prev_cost1 + new_cost1) / new_position1

        if new_position2 == 0:
            self.cost_basis2 = 0
        else:
            self.cost_basis2 = (prev_cost2 + new_cost2) / new_position2

        self.shares_held1 = new_position1
        self.shares_held2 = new_position2

        if self.position_change1 < 0:
            self.total_shares_sold += abs(self.position_change1)
            self.total_sales_value += abs(self.position_change1 * self.current_price1)
        if self.position_change2 < 0:
            self.total_shares_sold += abs(self.position_change2)
            self.total_sales_value += abs(self.position_change2 * self.current_price2)

        old_worth = self.net_worth

        self.net_worth = self.balance + self.shares_held1 * self.current_price1 + self.shares_held2 * self.current_price2
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        self.net_worth_change = self.net_worth - old_worth

    def step(self, action):
        initial_worth = self.net_worth

        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        self.t += 1

        # delay_modifier = (self.current_step / MAX_STEPS)
        #
        # reward = self.balance * delay_modifier
        reward = (self.net_worth - initial_worth) #/ initial_worth
        # reward = self.position_change1 * self.price_change1 + self.position_change2 * self.price_change2
        done = self.net_worth <= 0 or self.balance <= 0 or self.t == self.ep_len

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held1 = 0
        self.shares_held2 = 0
        self.cost_basis1 = 0
        self.cost_basis2 = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            self.lookback_num, len(self.df1.loc[:, 'Open'].values) - self.ep_len)
        self.t = 0

        self.current_price1 = random.uniform(
            self.df1.loc[self.current_step - 1, "High"], self.df1.loc[self.current_step - 1, "Low"])
        self.current_price2 = random.uniform(
            self.df2.loc[self.current_step - 1, "High"], self.df2.loc[self.current_step - 1, "Low"])

        # self.current_price1 = self.df1.loc[self.current_step - 1, "Close"]
        # self.current_price2 = self.df2.loc[self.current_step - 1, "Close"]

        self.price_change1 = 0
        self.price_change2 = 0

        return self._next_observation()

    def set_first(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held1 = 0
        self.shares_held2 = 0
        self.cost_basis1 = 0
        self.cost_basis2 = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = self.lookback_num
        self.t = 0

        self.current_price1 = random.uniform(
            self.df1.loc[self.current_step - 1, "Open"], self.df1.loc[self.current_step - 1, "Close"])
        self.current_price2 = random.uniform(
            self.df2.loc[self.current_step - 1, "Open"], self.df2.loc[self.current_step - 1, "Close"])

        self.current_price1 = self.df1.loc[self.current_step - 1, "Close"]
        self.current_price2 = self.df2.loc[self.current_step - 1, "Close"]

        self.price_change1 = 0
        self.price_change2 = 0

        return self._next_observation()

    # def render(self, mode='human', close=False):
    #     # Render the environment to the screen
    #     profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
    #
    #     print('Step: {}'.format(self.current_step))
    #     print('Balance: {}'.format(self.balance))
    #     print(
    #         'Shares held: {} (Total sold: {})'.format(self.shares_held,
    #                                                   self.total_shares_sold))
    #     print(
    #         'Avg cost for held shares: {} (Total sales value: {})'.format(self.cost_basis,
    #                                                                       self.total_sales_value))
    #     print(
    #         'Net worth: {} (Max net worth: {})'.format(self.net_worth,
    #                                                    self.max_net_worth))
    #     print('Profit: {}'.format(profit))
