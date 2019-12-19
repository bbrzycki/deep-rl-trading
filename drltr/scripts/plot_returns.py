import matplotlib as mpl
# mpl.use('agg')

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


INITIAL_ACCOUNT_BALANCE = 10000

def buy_and_hold(starting_balance, num_shares, close_data):
    net_worth = [starting_balance]
    remaining_balance = starting_balance - num_shares * close_data[0]
    for i in range(0, len(close_data)):
        net_worth.append(remaining_balance + num_shares * close_data[i])
    return np.array(net_worth)

def buy_and_hold2(starting_balance, num_shares, close_data, close_data2):
    net_worth = [starting_balance]
    remaining_balance = starting_balance - num_shares * close_data[0] - num_shares * close_data2[0]
    for i in range(0, len(close_data)):
        net_worth.append(remaining_balance + num_shares * close_data[i] + num_shares * close_data2[i])
    return np.array(net_worth)

def plot_one_equity(test_csv,
                    test_data_dir,
                    save_name='',
                    max_num_shares=3):
    if test_data_dir[-1] == '/':
        test_data_dir = test_data_dir[:-1]
    results_path = test_data_dir + '/test_net_worth.npy'
    results = np.load(results_path)

    close_array = pd.read_csv(test_csv)['Close']

    bah_results = buy_and_hold(INITIAL_ACCOUNT_BALANCE,
                               num_shares=max_num_shares,
                               close_data=close_array)

    fig = plt.figure()
    plt.plot(results, label='DQN')
    plt.plot(bah_results, label='Buy and Hold')

    plt.xlabel('Day')
    plt.ylabel('Net Worth')

    plt.legend()

    if save_name != '':
        plt.savefig(save_name, bbox_inches='tight')

    plt.show()

    print(results.shape, bah_results.shape)


def plot_two_equities(test_csv1,
                      test_csv2,
                      test_data_dir='',
                      save_name='',
                      max_num_shares=3):
    if test_data_dir != '':
        if test_data_dir[-1] == '/':
            test_data_dir = test_data_dir[:-1]
        results_path = test_data_dir + '/test_net_worth.npy'
        results = np.load(results_path)

    close_array1 = pd.read_csv(test_csv1)['Close']
    close_array2 = pd.read_csv(test_csv2)['Close']

    bah_results1 = buy_and_hold(INITIAL_ACCOUNT_BALANCE,
                                num_shares=max_num_shares,
                                close_data=close_array1)
    bah_results2 = buy_and_hold(INITIAL_ACCOUNT_BALANCE,
                                num_shares=max_num_shares,
                                close_data=close_array2)
    bah_both = buy_and_hold2(INITIAL_ACCOUNT_BALANCE,
                                num_shares=max_num_shares,
                                close_data=close_array1,
                                close_data2=close_array2)


    fig = plt.figure()
    if test_data_dir != '':
        plt.plot(results, label='DQN')
    plt.plot(bah_results1, '--', label='Buy and Hold 1')
    plt.plot(bah_results2, '--', label='Buy and Hold 2')
    plt.plot(bah_both, '--', label='Buy and Hold Both')

    plt.xlabel('Day')
    plt.ylabel('Net Worth')

    plt.legend()

    if save_name != '':
        plt.savefig(save_name, bbox_inches='tight')

    plt.show()
