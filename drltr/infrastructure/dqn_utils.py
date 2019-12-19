"""This file includes a collection of utility functions that are useful for
implementing DQN."""
import random
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from .tf_utils import lrelu

# from drltr.infrastructure.atari_wrappers import wrap_deepmind

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])


def get_env_kwargs(env_name, lookback_num=5, model='fc'):

    # if env_name == 'Discrete-1-Equity-v0':
    if env_name == 'Discrete-1-Equity-v0':
        def empty_wrapper(env):
            return env
        if model=='fc':
            model = trading_model
        elif model=='lstm':
            model = trading_model_lstm
        kwargs = {
            'optimizer_spec': trading_optimizer(),
            'q_func': model,
            'replay_buffer_size': 5000,
            'batch_size': 32,
            'gamma': 1,
            'learning_starts': 200,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 200,
            'grad_norm_clipping': 5,
            'num_timesteps': 2000,
            'env_wrappers': empty_wrapper,
            'lookback_num': lookback_num,
            'input_shape': (5, lookback_num + 2, 1),
        }
        kwargs['exploration_schedule'] = trading_exploration_schedule(kwargs['num_timesteps'])
    elif env_name == 'Discrete-1-Equity-Costs-v0':
        def empty_wrapper(env):
            return env
        if model=='fc':
            model = trading_model
        elif model=='lstm':
            model = trading_model_lstm
        kwargs = {
            'optimizer_spec': trading_optimizer(),
            'q_func': model,
            'replay_buffer_size': 5000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 200,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 200,
            'grad_norm_clipping': 5,
            'num_timesteps': 10000,
            'env_wrappers': empty_wrapper,
            'lookback_num': lookback_num,
            'input_shape': (5, lookback_num + 2, 1),
        }
        kwargs['exploration_schedule'] = trading_exploration_schedule(kwargs['num_timesteps'])
    elif env_name == 'Discrete-1-Equity-Short-v0':
        def empty_wrapper(env):
            return env
        if model=='fc':
            model = trading_model
        elif model=='lstm':
            model = trading_model_lstm
        kwargs = {
            'optimizer_spec': trading_optimizer(),
            'q_func': model,
            'replay_buffer_size': 5000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 200,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 200,
            'grad_norm_clipping': 5,
            'num_timesteps': 2000,
            'env_wrappers': empty_wrapper,
            'lookback_num': lookback_num,
            'input_shape': (5, lookback_num + 2, 1),
        }
        kwargs['exploration_schedule'] = trading_exploration_schedule(kwargs['num_timesteps'])
    elif env_name == 'Discrete-2-Equities-v0':
        def empty_wrapper(env):
            return env
        if model=='fc':
            model = trading_model
        elif model=='lstm':
            model = define_trading_model_lstm1(lookback_num)
        kwargs = {
            'optimizer_spec': trading_optimizer(),
            'q_func': model,
            'replay_buffer_size': 5000,
            'batch_size': 32,
            'gamma': 1.00,
            'learning_starts': 200,
            'learning_freq': 1,
            'frame_history_len': 1,
            'target_update_freq': 200,
            'grad_norm_clipping': 5,
            'num_timesteps': 2000,
            'env_wrappers': empty_wrapper,
            'lookback_num': lookback_num,
            'input_shape': (5, (lookback_num + 1) * 2 + 1, 1),
        }
        kwargs['exploration_schedule'] = trading_exploration_schedule(kwargs['num_timesteps'])

    else:
        raise NotImplementedError

    return kwargs


def trading_model(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def trading_model_lstm(obs, num_actions, scope, lookback_num=250, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        ts_out, aux_out = out[:, :, :-1, 0], out[:, :, -1:, 0]
        print(ts_out.shape, aux_out.shape)
        batch_size = tf.shape(ts_out)[0]

        cell = tf.nn.rnn_cell.LSTMCell(128, activation=tf.nn.tanh)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        lstm_out, state = tf.nn.dynamic_rnn(cell, tf.transpose(ts_out, perm=[0, 2, 1]),
                                            initial_state=initial_state, dtype=tf.float32)
        print(layers.flatten(lstm_out).shape, layers.flatten(aux_out).shape)
        out = tf.concat([layers.flatten(lstm_out), layers.flatten(aux_out)], axis=1)
        print(out.shape)
        out = layers.flatten(out)
        # out = tf.layers.batch_normalization(out)
        # out = lrelu(out)
        # out = layers.dropout(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            # out = layers.fully_connected(out, num_outputs=32, activation_fn=lrelu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def trading_model1(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


def define_trading_model_lstm1(lookback_num):
    def trading_model_lstm1(obs, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = obs
            ts_out1, ts_out2, aux_out = out[:, :, :lookback_num+1, 0], out[:, :, lookback_num+1:-1, 0],out[:, :, -1:, 0]
            batch_size = tf.shape(ts_out1)[0]
            # timesteps = tf.shape(ts_out)[2]
            ts_out = tf.concat([ts_out1, ts_out2], axis=1)
            print(ts_out.shape, aux_out.shape)

            cell = tf.nn.rnn_cell.LSTMCell(8, activation=tf.nn.tanh)
            initial_state = cell.zero_state(batch_size, dtype=tf.float32)
            lstm_out, state = tf.nn.dynamic_rnn(cell, tf.transpose(ts_out, perm=[0, 2, 1]),
                                                initial_state=initial_state, dtype=tf.float32)
            print(layers.flatten(lstm_out).shape, layers.flatten(aux_out).shape)
            out = tf.concat([layers.flatten(lstm_out), layers.flatten(aux_out)], axis=1)
            print(out.shape)
            out = layers.flatten(out)
            # out = tf.layers.batch_normalization(out)
            # out = lrelu(out)
            # out = layers.dropout(out)
            with tf.variable_scope("action_value"):
                out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
                # out = layers.fully_connected(out, num_outputs=32, activation_fn=tf.nn.relu)
                # out = layers.fully_connected(out, num_outputs=64, activation_fn=lrelu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out
    return trading_model_lstm1

def trading_optimizer():
    return OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-4),
        kwargs={}
    )


def trading_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.4),
            (num_timesteps * 0.3, 0.2)
        ], outside_value=0.01
    )


def lander_model(obs, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = obs
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=64, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out


# Allow for some customization of the network
def make_lander_model(arch=(64, 64)):
    def lander_model(obs, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = obs
            with tf.variable_scope("action_value"):
                out = layers.fully_connected(out, num_outputs=arch[0], activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=arch[1], activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

            return out
    return lander_model


def atari_model(img_input, num_actions, scope, reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        out = tf.cast(img_input, tf.float32) / 255.0
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_ram_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 0.2),
            (1e6, 0.1),
            (num_timesteps / 8, 0.01),
        ], outside_value=0.01
    )


def atari_optimizer(num_timesteps):
    num_iterations = num_timesteps/4
    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2, 5e-5 * lr_multiplier),
    ],
        outside_value=5e-5 * lr_multiplier)

    return OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )


def lander_optimizer():
    return OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        lr_schedule=ConstantSchedule(1e-3),
        kwargs={}
    )


def lander_exploration_schedule(num_timesteps):
    return PiecewiseSchedule(
        [
            (0, 1),
            (num_timesteps * 0.1, 0.02),
        ], outside_value=0.02
    )


def huber_loss(x, delta=1.0):
    # https://en.wikipedia.org/wiki/Huber_loss
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )


def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns
    comparable objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):
    def __init__(self, value):
        """Value remains constant over time.
        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

def compute_exponential_averages(variables, decay):
    """Given a list of tensorflow scalar variables
    create ops corresponding to their exponential
    averages
    Parameters
    ----------
    variables: [tf.Tensor]
        List of scalar tensors.
    Returns
    -------
    averages: [tf.Tensor]
        List of scalar tensors corresponding to averages
        of al the `variables` (in order)
    apply_op: tf.runnable
        Op to be run to update the averages with current value
        of variables.
    """
    averager = tf.train.ExponentialMovingAverage(decay=decay)
    apply_op = averager.apply(variables)
    return [averager.average(v) for v in variables], apply_op

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)

def initialize_interdependent_variables(session, vars_list, feed_dict):
    """Initialize a list of variables one at a time, which is useful if
    initialization of some variables depends on initialization of the others.
    """
    vars_left = vars_list
    while len(vars_left) > 0:
        new_vars_left = []
        for v in vars_left:
            try:
                session.run(tf.variables_initializer([v]), feed_dict)
            except tf.errors.FailedPreconditionError:
                new_vars_left.append(v)
        if len(new_vars_left) >= len(vars_left):
            # This can happen if the variables all depend on each other, or more likely if there's
            # another variable outside of the list, that still needs to be initialized. This could be
            # detected here, but life's finite.
            raise Exception("Cycle in variable dependencies, or extenrnal precondition unsatisfied.")
        else:
            vars_left = new_vars_left

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)

class MemoryOptimizedReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """

        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            # print(self.obs[start_idx:end_idx].shape)
            # print(img_h, img_w)
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)
            # return self.obs[start_idx:end_idx].transpose(1, 2, 0).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.float32)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after observing frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done
