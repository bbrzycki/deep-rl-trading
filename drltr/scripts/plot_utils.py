import matplotlib as mpl
# mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def parse_tf_events_file(filename, key='Eval_AverageReturn'):
    eval_returns = []
    for e in tf.train.summary_iterator(filename):
        for v in e.summary.value:
            if v.tag == key:
                eval_returns.append(v.simple_value)

    return eval_returns


def plot_fn_labels(fn_labels, plt_fn=None, figsize=None):
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=figsize)

    for filename, label, ls in fn_labels:
        plt.plot(parse_tf_events_file(filename), label=label, ls=ls)

    plt.xlabel('Iteration')
    plt.ylabel('Average Return')

    plt.grid()
    plt.legend()

    if plt_fn is not None:
        plt.savefig(plt_fn, bbox_inches='tight')
    plt.show()
