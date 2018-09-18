from typing import Mapping, Callable
from functools import partial
import tensorflow as tf
import numpy as np
from .ddpg import qlearning_loss_term


def random_shuffle(*args, axis=1):
    """
    >>> tf.set_random_seed(0)
    >>> a = tf.reshape(tf.range(6), (3,2))
    >>> b = tf.reshape(tf.range(9), (3,3))
    >>> with tf.Session() as sess:
    >>>     random_shuffle(a, b, axis=1)
    """
    size_splits = [int(a.shape[axis]) for a in args]
    cat_args = tf.concat(args, axis=axis)
    shuff_cat_args = tf.random_shuffle(cat_args)    # axis=0
    return tf.split(shuff_cat_args, num_or_size_splits=size_splits, axis=axis)


def lower_bound_loss_term_fwrl(batch_tf: Mapping[str, tf.Tensor],
                               target_net_fn: Callable,
                               main_net_fn: Callable,
                               post_process_target_ret: Callable=lambda x: x,
                               gamma=None):
    """Returns a loss term as a function of inputs reusing the variables from
       variable_scope

    Impelements the loss function
    L = | bounded(Qₜ(o, g', u) + γ Qₜ(o', g, μ(o', g))) - Qₘ(o, g, u) |²₊
    """
    # Check inputs for o, ag, g, reward
    for k in 'o ag g r'.split():
        assert k in batch_tf, "no key {} in batch_tf {}".format(
            k, batch_tf.keys())

    # compute permutations of available observations and the corresponding goal
    # locations
    # NOTE: Q: Does FW equation assumes same reward function independent of
    #          goal?
    #       A: Yes
    #       Q: Will FW work if the reward function is say smooth from the
    #          distance towards the goal?
    #       A: No.
    # TODO: Think more about it^^
    o_i, g_i = random_shuffle(batch_tf['o'], batch_tf['ag'], axis=1)

    Q_via_i = (
        # Qₜ(o, g', u)
        target_net_fn(
            inputs=dict(o=batch_tf['o'], g=g_i, u=batch_tf['u']))
        .Q_tf +
        # How's the max being computed over u?
        # Because we are chosing the Q_pi_tf
        # Qₜ(o', g, μ(o', g))
        gamma * target_net_fn(inputs=dict(o=o_i, g=batch_tf['g'], u=batch_tf['u']))
        .Q_pi_tf)
    Q_via_i_clipped = post_process_target_ret(Q_via_i)

    # Qₘ(o, g, u)
    Q_best = main_net_fn(
        inputs=dict(o=batch_tf['o'], g=batch_tf['g'], u=batch_tf['u'])
    ).Q_tf
    return tf.reduce_mean(
        tf.square(
            tf.nn.relu(
                tf.stop_gradient(Q_via_i_clipped) - Q_best)))


def upper_bound_loss_term_fwrl(batch_tf: Mapping[str, tf.Tensor],
                               target_net_fn: Callable,
                               main_net_fn: Callable,
                               post_process_target_ret: Callable=lambda x: x,
                               gamma=None):
    """Returns a loss term as a function of inputs reusing the variables from
       variable_scope

    Impelements the loss function
    L = | Qₘ(o, g', u) + γ Qₘ(o', g, μ(o', g)) - bounded(Qₜ(o, g, u)) |²₊
    """
    # Check inputs for o, ag, g, reward
    for k in 'o ag g r'.split():
        assert k in batch_tf, "no key {} in batch_tf {}".format(
            k, batch_tf.keys())

    o_i, g_i = random_shuffle(batch_tf['o'], batch_tf['ag'], axis=1)

    Q_via_i = (
        # Qₘ(o, g', u)
        main_net_fn(
            inputs=dict(o=batch_tf['o'], g=g_i, u=batch_tf['u']))
        .Q_tf +
        # How's the max being computed over u?
        # Because we are chosing the Q_pi_tf
        # Qₘ(o', g, μ(o', g))
        gamma * main_net_fn(inputs=dict(o=o_i, g=batch_tf['g'], u=batch_tf['u']))
        .Q_pi_tf)

    # Qₜ(o, g, u)
    Q_best = target_net_fn(
        inputs=dict(o=batch_tf['o'], g=batch_tf['g'], u=batch_tf['u'])
    ).Q_tf
    Q_best_clipped = post_process_target_ret(Q_best)
    return tf.reduce_mean(
        tf.square(
            tf.nn.relu(
                tf.stop_gradient(Q_via_i) - Q_best_clipped)))


def step_loss_term_fwrl(batch_tf: Mapping[str, tf.Tensor],
                        target_net_fn: Callable,
                        main_net_fn: Callable,
                        post_process_target_ret: Callable=lambda x: x,
                        gamma=None):
    main = main_net_fn(inputs=dict(o=batch_tf['o'],
                                   g=batch_tf['ag_2'],
                                   u=batch_tf['u']))
    loss = tf.reduce_mean(tf.square(batch_tf['r'] - main.Q_tf))
    return loss


def sum_loss_terms(batch_tf: Mapping[str, tf.Tensor],
                   target_net_fn: Callable,
                   main_net_fn: Callable,
                   loss_terms_fns = [],
                   **kwargs):
    return sum(f(batch_tf, target_net_fn, main_net_fn, **kwargs)
               for f in loss_terms_fns)


bounds_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [lower_bound_loss_term_fwrl, upper_bound_loss_term_fwrl])


qlearning_constrained_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [qlearning_loss_term, lower_bound_loss_term_fwrl,
                      upper_bound_loss_term_fwrl])


step_with_constraint_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [lower_bound_loss_term_fwrl, step_loss_term_fwrl,
                      upper_bound_loss_term_fwrl])
