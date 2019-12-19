import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 100000000
MAX_NUM_SHARES = 100000000
MAX_SHARE_PRICE = 1000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000




class Discrete1EquityCosts(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(Discrete1EquityCosts, self).__init__()

        # self.df = df
        # self.ep_len = ep_len

        self.lookback_num = 5

        self.reward_range = (0, MAX_ACCOUNT_BALANCE)


        self.action_space = spaces.Discrete(4)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, self.lookback_num + 2), dtype=np.float16)

    def set_init(self, df, ep_len, lookback_num):
        self.df = df
        self.ep_len = ep_len
        self.lookback_num = lookback_num
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, self.lookback_num + 2), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step - self.lookback_num:self.current_step, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.lookback_num:self.current_step, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.lookback_num:self.current_step, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.lookback_num:self.current_step, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step - self.lookback_num:self.current_step, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # print(frame)
        # print(self.current_step)

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, np.array([[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / 3,
            self.cost_basis / MAX_SHARE_PRICE,
            # self.total_shares_sold / MAX_NUM_SHARES,
            # self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
            (self.price_change + 1) / 100,
        ]]).transpose(), axis=1)

        return obs[:, :, np.newaxis]

    def _take_action(self, action):
        self.last_price = self.current_price

        # Set the current price to a random price within the time step
        self.current_price = random.uniform(
            self.df.loc[self.current_step, "High"], self.df.loc[self.current_step, "Low"])
        # self.current_price = self.df.loc[self.current_step, "Close"]

        self.price_change = (self.current_price - self.last_price) / self.last_price

        # Define how action values are mapped to actual actions
        new_position = action #- 3
        # self.position_change = action - 3
        # new_position = self.shares_held + self.position_change

        self.position_change = new_position - self.shares_held
        prev_cost = self.cost_basis * self.shares_held
        new_cost = self.position_change * self.current_price

        self.balance -= new_cost

        # transaction cost
        if self.position_change != 0:
            self.balance -= 9

        if new_position == 0:
            self.cost_basis = 0
        else:
            self.cost_basis = (prev_cost + new_cost) / new_position

        self.shares_held = new_position


        if self.position_change < 0:
            self.total_shares_sold += abs(self.position_change )
            self.total_sales_value += abs(self.position_change  * self.current_price)

        old_worth = self.net_worth

        self.net_worth = self.balance + self.shares_held * self.current_price
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
        reward = (self.net_worth - initial_worth) #/ self.last_price
        # reward = self.net_worth

        # reward = self.price_change * self.last_price * (self.shares_held - self.position_change)
        # Penalize for wrong moves
        # if self.price_change < 0 and self.position_change >= 0:
        #         reward += self.price_change * self.max_num_shares
        # elif self.price_change > 0 and self.position_change <= 0:
        #         reward -= self.price_change * self.max_num_shares
        #
        # if self.position_change == 0:
        #     reward -= 0.1

        done = self.net_worth <= 0 or self.balance <= 0 or self.t == self.ep_len

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
            self.lookback_num, len(self.df.loc[:, 'Open'].values) - self.ep_len)
        self.t = 0

        self.current_price = random.uniform(
            self.df.loc[self.current_step - 1, "High"], self.df.loc[self.current_step - 1, "Low"])
        # self.current_price = self.df.loc[self.current_step, "Close"]

        self.price_change = 0

        return self._next_observation()

    def set_first(self):
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = self.lookback_num
        self.t = 0

        self.current_price = random.uniform(
            self.df.loc[self.current_step - 1, "High"], self.df.loc[self.current_step - 1, "Low"])
        self.current_price = self.df.loc[self.current_step - 1, "Close"]

        self.price_change = 0

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
