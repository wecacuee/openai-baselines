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
    #o_i, g_i = random_shuffle(batch_tf['o'], batch_tf['ag'], axis=1)
    o_i = batch_tf['o_i']
    g_i = batch_tf['g_i']

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

    #o_i, g_i = random_shuffle(batch_tf['o'], batch_tf['ag'], axis=1)
    o_i = batch_tf['o_i']
    g_i = batch_tf['g_i']

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
"""
This should work because of log term perspective but needs HER sampling.
"""

step_with_constraint_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [lower_bound_loss_term_fwrl, step_loss_term_fwrl,
                      upper_bound_loss_term_fwrl])
"""
This should work without HER sampling because of one step reward consideration
"""


qlearning_step_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [qlearning_loss_term, step_loss_term_fwrl])
"""
Experiments show that step term does better than step + constraints. Does step
+ qlearning does better than just qlearning?
"""


qlearning_step_constrained_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [qlearning_loss_term,
                      lower_bound_loss_term_fwrl,
                      step_loss_term_fwrl,
                      upper_bound_loss_term_fwrl])
"""
This might work because of a balance between HER and one-step reward.
"""


step_lower_bound_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [lower_bound_loss_term_fwrl, step_loss_term_fwrl])
"""
Lower bound with step
"""


step_upper_bound_loss_term_fwrl = partial(
    sum_loss_terms,
    loss_terms_fns = [upper_bound_loss_term_fwrl, step_loss_term_fwrl])
"""
Upper bound with step
"""


def _sample_fwrl_transitions(episode_batch, batch_size_in_transitions,
                             future_p=None, reward_fun=None):
    """episode_batch is {key: array(buffer_size x T x dim_key)}
    """
    assert future_p is not None
    assert reward_fun is not None
    T = episode_batch['u'].shape[1]
    rollout_batch_size = episode_batch['u'].shape[0]
    batch_size = batch_size_in_transitions

    # Select which episodes and time steps to use.
    episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
    t_samples = np.random.randint(T, size=batch_size)
    transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
                   for key in episode_batch.keys()}

    # Select future time indexes proportional with probability future_p. These
    # will be used for HER replay by substituting in future goals.
    her_index_mask = np.random.uniform(size=batch_size) < future_p
    her_indexes = np.where(her_index_mask)
    future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
    future_offset = future_offset.astype(int)
    future_all_t = (t_samples + 1 + future_offset)
    future_t = future_all_t[her_indexes]
    future_all_t[~her_index_mask] = T-1

    # Replace goal with achieved goal but only for the previously-selected
    # HER transitions (as defined by her_indexes). For the other transitions,
    # keep the original goal.
    # NOTE: HER transitions g <- ag
    future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
    transitions['g'][her_indexes] = future_ag

    # Add intermediate goal information for FWRL
    imdt_samples = np.random.rand(batch_size)
    intermediate_t = (t_samples * (1-imdt_samples)
                      + future_all_t * imdt_samples).astype(int)
    transitions['ag_im'] = episode_batch['ag'][episode_idxs, intermediate_t]
    transitions['o_im'] = episode_batch['o'][episode_idxs, intermediate_t]

    # Reconstruct info dictionary for reward  computation.
    info = {}
    for key, value in transitions.items():
        if key.startswith('info_'):
            info[key.replace('info_', '')] = value

    # Re-compute reward since we may have substituted the goal.
    reward_params = {k: transitions[k] for k in ['ag_2', 'g']}
    reward_params['info'] = info
    transitions['r'] = reward_fun(**reward_params)

    transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:])
                   for k in transitions.keys()}

    assert(transitions['u'].shape[0] == batch_size_in_transitions)

    return transitions


def make_sample_fwrl_transitions(replay_strategy, replay_k, reward_fun):
    """Creates a sample function that can be used for FWRL experience replay.
    It samples intermediate states as part of experience.

    TODO: Sample an intermediate state that is exactly mid way between t and
    future_t. Optimize that way.
    TODO: Is this like Binary search but for reinforcement learning?

    Args:
        replay_strategy (in ['future', 'none']): the HER replay strategy; if
        set to 'none', regular DDPG experience replay is used

        replay_k (int): the ratio between HER replays and regular replays (e.g. k = 4 -> 4 times
            as many HER replays as regular replays are used)
        reward_fun (function): function to re-compute the reward with substituted goals
    """
    if replay_strategy == 'future':
        future_p = 1 - (1. / (1 + replay_k))
    else:  # 'replay_strategy' == 'none'
        future_p = 0

    return partial(_sample_fwrl_transitions, future_p = future_p,
                   reward_fun=reward_fun)

