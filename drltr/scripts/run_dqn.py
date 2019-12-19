import os
import time

import pandas as pd

from drltr.infrastructure.rl_trainer import RL_Trainer
from drltr.agents.dqn_agent import DQNAgent
from drltr.infrastructure.dqn_utils import get_env_kwargs


class Q_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'],
                                  self.params['lookback_num'],
                                  self.params['model'])

        self.agent_params = {**train_args, **env_args, **params}

        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = self.agent_params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = self.agent_params['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.agent_params['num_timesteps'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

    def run_test(self):
        self.rl_trainer.run_test()

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name',  default='Discrete-1-Equity-v0',
                        choices=('Discrete-1-Equity-v0',
                                 'Discrete-1-Equity-Short-v0',
                                 'Discrete-1-Equity-Costs-v0',
                                 'Discrete-2-Equities-v0')
                        )

    parser.add_argument('--ep_len', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='todo')

    parser.add_argument('--eval_batch_size', type=int, default=100)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1)
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--double_q', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    parser.add_argument('--scalar_log_freq', type=int, default=1)

    parser.add_argument('--save_params', action='store_true')

    # parser.add_argument('--save_sess', '-s', action='store_true')
    parser.add_argument('--save_sess_freq', type=int, default=-1)
    parser.add_argument('--load_sess', '-l', type=str, default='')

    parser.add_argument('--train_data_path', '-train', type=str, default='')
    parser.add_argument('--train_data_path2', '-train2', type=str, default='')

    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--test_data_path', '-test', type=str, default='')
    parser.add_argument('--test_data_path2', '-test2', type=str, default='')

    parser.add_argument('--lookback_num', '-lb', type=int, default=5)
    parser.add_argument('--model', '-m', type=str, default='fc')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)
    params['video_log_freq'] = -1 # This param is not used for DQN

    # Put data in df format for environment
    if params['train_data_path'] != '':
        params['train_df'] = pd.read_csv(params['train_data_path']).sort_values('Date')
    if params['test_data_path'] != '':
        params['test_df'] = pd.read_csv(params['test_data_path']).sort_values('Date')

    if params['train_data_path2'] != '':
        params['train_df2'] = pd.read_csv(params['train_data_path2']).sort_values('Date')
    if params['test_data_path2'] != '':
        params['test_df2'] = pd.read_csv(params['test_data_path2']).sort_values('Date')

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    logdir_prefix = 'dqn_'
    if args.double_q:
        logdir_prefix += 'double_q_'

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    # Commenting out date for now for ease of plotting...
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name #+ '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    trainer = Q_Trainer(params)
    if params['train_data_path'] != '' and not params['only_test']:
        trainer.run_training_loop()
    if params['test_data_path'] != '':
        trainer.run_test()


if __name__ == "__main__":
    main()
