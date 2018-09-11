from typing import Mapping, Callable
import tensorflow as tf


def random_shuffle(*args, axis=1):
    """
    >>> tf.set_random_seed(0)
    >>> a = tf.reshape(tf.range(6), (3,2))
    >>> b = tf.reshape(tf.range(9), (3,3))
    >>> with tf.Session() as sess:
    >>>     random_shuffle(a, b, axis=1)
    """
    size_splits = [a.shape[axis] for a in args]
    cat_args = tf.concat(args, axis=axis)
    shuff_cat_args = tf.random_shuffle(cat_args)    # axis=0
    return tf.split(shuff_cat_args, num_or_size_splits=size_splits, axis=axis)


def addnl_loss_term_fwrl(batch_tf: Mapping[str, tf.Tensor],
                         target_net_fn: Callable,
                         main_net_fn: Callable):
    """Returns a loss term as a function of inputs reusing the variables from
       variable_scope

    """
    # Check inputs for o, ag, g, reward
    for k in 'o ag g r'.split():
        assert k in batch_tf, "no key {} in batch_tf".format(k)

    # compute permutations of available observations and the corresponding goal
    # locations
    # NOTE: Q: Does FW equation assumes same reward function independent of
    #          goal?
    #       A: Yes
    #       Q: Will FW work if the reward function is say smooth from the
    #          distance towards the goal?
    #       A: No.
    # TODO: Think more about it^^
    assert batch_tf['o'].shape[0] == batch_tf['ag'].shape[0]
    assert batch_tf['o'].shape[1] == batch_tf['ag'].shape[1]
    o_i, g_i = random_shuffle(batch_tf['o'], batch_tf['ag'], axis=2)

    Q_via_i = (
        target_net_fn(
            inputs=dict(o=batch_tf['o'], g=g_i, u=batch_tf['u']))
        .Q_tf +
        target_net_fn(inputs=dict(o=o_i, g=batch_tf['g']))
        .Q_pi_tf)

    Q_best = main_net_fn(
        inputs=dict(batch_tf['o'], batch_tf['g'], u=batch_tf['u'])
    ).Q_tf
    return tf.reduce_mean(
        tf.square(
            tf.nn.relu(
                tf.stop_gradient(Q_via_i) - Q_best)))
