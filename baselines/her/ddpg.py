from functools import partial
from collections import OrderedDict
from typing import Mapping, Callable

import numpy as np
import tensorflow as tf
from tensorflow.contrib.staging import StagingArea

from baselines import logger
from baselines.her.util import (
    import_function, store_args, flatten_grads, transitions_in_episode_batch)
from baselines.her.normalizer import Normalizer
from baselines.her.replay_buffer import ReplayBuffer
from baselines.common.mpi_adam import MpiAdam


def dims_to_shapes(input_dims):
    return {key: tuple([val]) if val > 0 else tuple() for key, val in input_dims.items()}


def loss_term_noop(inputs, target_net_fn, main_net_fn):
    """Returns a loss term as a function of inputs reusing the variables from
       variable_scope
    """
    # NOTE: does NOT works with
    # return tf.zeros((0,))
    # NOTE: does NOT works with
    # return 0
    # NOTE: works with
    return tf.constant(0.0)
    # NOTE: works with
    # return tf.zeros((1,))


def qlearning_loss_term(batch_tf: Mapping[str, tf.Tensor],
                        target_net_fn: Callable,
                        main_net_fn: Callable,
                        post_process_target_ret: Callable=lambda x: x,
                        gamma=None,
                        return_main_target=True):
    # networks
    main = main_net_fn(inputs=batch_tf)
    # NOTE: What is o_2 and g_2. Why replace inputs for target network?
    # NOTE: o_2 and g_2 are just next step observations and next step
    # goals?
    # o_2 = o_{t+1} if o = o_{t}
    # NOTE: If we understand o_2 and g_2 right, then why target
    # networks get them as inputs?
    # Oh !! because of the bellman equation where the target network
    # gets evaluated on the next state. It is interesting that we need
    # g_2 for that.
    target = target_net_fn(inputs=dict(o=batch_tf['o_2'],
                                       g=batch_tf['g_2'],
                                       u=batch_tf['u']))

    # loss functions
    # Qₜ(s', μ(s',a'))
    target_Q_pi_tf = target.Q_pi_tf
    # target_tf = r + γ Qₜ(s', μ(s',a'))
    target_tf = post_process_target_ret(batch_tf['r'] + gamma * target_Q_pi_tf)
    # Q_loss_tf = (r + γ Qₜ(s', μ(s', a')) - Qₒ(s, δ(a)))²
    Q_loss_tf = tf.reduce_mean(
        tf.square(tf.stop_gradient(target_tf) - main.Q_tf))
    return Q_loss_tf


def with_scope_create_net(net_creator, inputs, variable_scope="", reuse=False):
    with tf.variable_scope(variable_scope) as vs:
        if reuse:
            vs.reuse_variables()
        ret = net_creator(inputs)
        vs.reuse_variables()
        return ret


def stage_shapes_frm_input_dims(input_dims):
    # Prepare staging area for feeding data to the model.
    # NOTE: input_dims.keys() are ['o', 'g', 'u', 'info_*' ...]

    input_shapes = dims_to_shapes(input_dims)
    stage_shapes = OrderedDict()
    for key in sorted(input_dims.keys()):
        if key.startswith('info_'):
            continue
        stage_shapes[key] = (None, *input_shapes[key])
    for key in ['o', 'g']:
        stage_shapes[key + '_2'] = stage_shapes[key]
    stage_shapes['r'] = (None,)
    stage_shapes['ag'] = (None, *input_shapes['g'])
    stage_shapes['ag_2'] = (None, *input_shapes['g'])
    stage_shapes['ag_im'] = (None, *input_shapes['g'])
    stage_shapes['o_im'] = (None, *input_shapes['o'])
    return stage_shapes


class DDPG(object):
    @store_args
    def __init__(self, input_dims, buffer_size, hidden, layers, network_class, polyak, batch_size,
                 Q_lr, pi_lr, norm_eps, norm_clip, max_u, action_l2, clip_obs, scope, T,
                 rollout_batch_size, subtract_goals, relative_goals, clip_pos_returns, clip_return,
                 sample_transitions, gamma, reuse=False,
                 loss_term=qlearning_loss_term,
                 **kwargs):
        """Implementation of DDPG that is used in combination with Hindsight Experience Replay (HER).

        Args:
            input_dims (dict of ints): dimensions for the observation (o), the goal (g), and the
                actions (u)
            buffer_size (int): number of transitions that are stored in the replay buffer
            hidden (int): number of units in the hidden layers
            layers (int): number of hidden layers
            network_class (str): the network class that should be used (e.g. 'baselines.her.ActorCritic')
            polyak (float): coefficient for Polyak-averaging of the target network
            batch_size (int): batch size for training
            Q_lr (float): learning rate for the Q (critic) network
            pi_lr (float): learning rate for the pi (actor) network
            norm_eps (float): a small value used in the normalizer to avoid numerical instabilities
            norm_clip (float): normalized inputs are clipped to be in [-norm_clip, norm_clip]
            max_u (float): maximum action magnitude, i.e. actions are in [-max_u, max_u]
            action_l2 (float): coefficient for L2 penalty on the actions
            clip_obs (float): clip observations before normalization to be in [-clip_obs, clip_obs]
            scope (str): the scope used for the TensorFlow graph
            T (int): the time horizon for rollouts
            rollout_batch_size (int): number of parallel rollouts per DDPG agent
            subtract_goals (function): function that subtracts goals from each other
            relative_goals (boolean): whether or not relative goals should be fed into the network
            clip_pos_returns (boolean): whether or not positive returns should be clipped
            clip_return (float): clip returns to be in [-clip_return, clip_return]
            sample_transitions (function) function that samples from the replay buffer
            gamma (float): gamma used for Q learning updates
            reuse (boolean): whether or not the networks should be reused
        """
        if self.clip_return is None:
            self.clip_return = np.inf

        self.create_actor_critic = import_function(self.network_class)

        input_shapes = dims_to_shapes(self.input_dims)
        self.dimo = self.input_dims['o']
        self.dimg = self.input_dims['g']
        self.dimu = self.input_dims['u']

        stage_shapes = stage_shapes_frm_input_dims(self.input_dims)
        self.stage_shapes = stage_shapes

        # Create network.
        with tf.variable_scope(self.scope):
            self.staging_tf = StagingArea(
                dtypes=[tf.float32 for _ in self.stage_shapes.keys()],
                shapes=list(self.stage_shapes.values()))
            self.buffer_ph_tf = [
                tf.placeholder(tf.float32, shape=shape)
                for shape in self.stage_shapes.values()]
            self.stage_op = self.staging_tf.put(self.buffer_ph_tf)

            self._create_network(reuse=reuse)

        # Configure the replay buffer.
        buffer_shapes = {key: (self.T if key != 'o' else self.T+1, *input_shapes[key])
                         for key, val in input_shapes.items()}
        buffer_shapes['g'] = (buffer_shapes['g'][0], self.dimg)
        buffer_shapes['ag'] = (self.T+1, self.dimg)

        buffer_size = (self.buffer_size // self.rollout_batch_size) * self.rollout_batch_size
        self.buffer = ReplayBuffer(buffer_shapes, buffer_size, self.T, self.sample_transitions)

        self.log_critic_loss = 0
        self.log_actor_loss = 0

    def _random_action(self, n):
        return np.random.uniform(low=-self.max_u, high=self.max_u, size=(n, self.dimu))

    def _preprocess_og(self, o, ag, g):
        # NOTE: goal is g <- g - ag
        # conditioned on relative_goals which is False in DEFAULT_PARAMS
        # NOTE: ag is ignored if relative_goals is False
        if self.relative_goals:
            g_shape = g.shape
            g = g.reshape(-1, self.dimg)
            ag = ag.reshape(-1, self.dimg)
            g = self.subtract_goals(g, ag)
            g = g.reshape(*g_shape)
        o = np.clip(o, -self.clip_obs, self.clip_obs)
        g = np.clip(g, -self.clip_obs, self.clip_obs)
        return o, g

    def get_actions(self, o, ag, g, noise_eps=0., random_eps=0., use_target_net=False,
                    compute_Q=False):
        o, g = self._preprocess_og(o, ag, g)
        policy = self.target if use_target_net else self.main
        # values to compute
        vals = [policy.pi_tf]
        if compute_Q:
            vals += [policy.Q_pi_tf]
        # feed
        feed = {
            policy.o_tf: o.reshape(-1, self.dimo),
            policy.g_tf: g.reshape(-1, self.dimg),
            policy.u_tf: np.zeros((o.size // self.dimo, self.dimu), dtype=np.float32)
        }

        ret = self.sess.run(vals, feed_dict=feed)
        # action postprocessing
        u = ret[0]
        noise = noise_eps * self.max_u * np.random.randn(*u.shape)  # gaussian noise
        u += noise
        u = np.clip(u, -self.max_u, self.max_u)
        u += np.random.binomial(1, random_eps, u.shape[0]).reshape(-1, 1) * (self._random_action(u.shape[0]) - u)  # eps-greedy
        if u.shape[0] == 1:
            u = u[0]
        u = u.copy()
        ret[0] = u

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def store_episode(self, episode_batch, update_stats=True):
        """
        episode_batch: array of batch_size x (T or T+1) x dim_key
                       'o' is of size T+1, others are of size T
        """

        self.buffer.store_episode(episode_batch)

        if update_stats:
            # add transitions to normalizer
            episode_batch['o_2'] = episode_batch['o'][:, 1:, :]
            episode_batch['ag_2'] = episode_batch['ag'][:, 1:, :]
            num_normalizing_transitions = transitions_in_episode_batch(episode_batch)
            transitions = self.sample_transitions(episode_batch, num_normalizing_transitions)

            o, o_2, g, ag = transitions['o'], transitions['o_2'], transitions['g'], transitions['ag']
            transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
            # No need to preprocess the o_2 and g_2 since this is only used for stats

            self.o_stats.update(transitions['o'])
            self.g_stats.update(transitions['g'])

            self.o_stats.recompute_stats()
            self.g_stats.recompute_stats()

    def get_current_buffer_size(self):
        return self.buffer.get_current_size()

    def _sync_optimizers(self):
        self.Q_adam.sync()
        self.pi_adam.sync()

    def _grads(self):
        # Avoid feed_dict here for performance!
        critic_loss, actor_loss, Q_grad, pi_grad = self.sess.run([
            self.Q_loss_tf,
            self.main.Q_pi_tf,
            self.Q_grad_tf,
            self.pi_grad_tf
        ])
        self.log_critic_loss = critic_loss
        self.log_actor_loss = actor_loss
        return critic_loss, actor_loss, Q_grad, pi_grad

    def _update(self, Q_grad, pi_grad):
        self.Q_adam.update(Q_grad, self.Q_lr)
        self.pi_adam.update(pi_grad, self.pi_lr)

    def sample_batch(self):
        # NOTE: Q: What are keys in transitions?
        #       A: ['r', 'o_2', 'ag_2', 'ag', 'g', 'o']
        transitions = self.buffer.sample(self.batch_size)
        o, o_2, g = transitions['o'], transitions['o_2'], transitions['g']
        ag, ag_2 = transitions['ag'], transitions['ag_2']
        transitions['o'], transitions['g'] = self._preprocess_og(o, ag, g)
        transitions['o_2'], transitions['g_2'] = self._preprocess_og(o_2, ag_2, g)
        o_im, ag_im = transitions['o_im'], transitions['ag_im']
        transitions['o_im'], transitions['ag_im'] = self._preprocess_og(o_im, ag_im, g)

        # NOTE: The stage_shapes.keys() are ['r', 'u', 'o', 'o_2', 'g', 'g_2']
        transitions_batch = [transitions[key] for key in self.stage_shapes.keys()]
        return transitions_batch

    def stage_batch(self, batch=None):
        if batch is None:
            batch = self.sample_batch()
        assert len(self.buffer_ph_tf) == len(batch)
        self.sess.run(self.stage_op, feed_dict=dict(zip(self.buffer_ph_tf, batch)))

    def train(self, stage=True):
        if stage:
            self.stage_batch()
        critic_loss, actor_loss, Q_grad, pi_grad = self._grads()
        self._update(Q_grad, pi_grad)
        return critic_loss, actor_loss

    def _init_target_net(self):
        self.sess.run(self.init_target_net_op)

    def update_target_net(self):
        self.sess.run(self.update_target_net_op)

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def _create_network(self, reuse=False):
        logger.info("Creating a DDPG agent with action space %d x %s..." % (self.dimu, self.max_u))

        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        # running averages
        with tf.variable_scope('o_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.o_stats = Normalizer(self.dimo, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('g_stats') as vs:
            if reuse:
                vs.reuse_variables()
            self.g_stats = Normalizer(self.dimg, self.norm_eps, self.norm_clip, sess=self.sess)

        # mini-batch sampling.
        batch = self.staging_tf.get()
        batch_tf = OrderedDict([(key, batch[i])
                                for i, key in enumerate(self.stage_shapes.keys())])
        batch_tf['r'] = tf.reshape(batch_tf['r'], [-1, 1])

        main_net_fn = partial(
            with_scope_create_net,
            net_creator=partial(self.create_actor_critic,
                                net_type='main', **self.__dict__),
            variable_scope="main",
            reuse=reuse)
        self.main = main_net_fn(inputs=batch_tf)
        target_net_fn = partial(
            with_scope_create_net,
            net_creator=partial(self.create_actor_critic,
                                net_type='target', **self.__dict__),
            variable_scope="target",
            reuse=reuse)
        self.target = target_net_fn(inputs=dict(o=batch_tf['o_2'],
                                                g=batch_tf['g_2'],
                                                u=batch_tf['u']))

        # NOTE: Does the AddnlLossTerm gets added to both pi_loss_tf and
        # Q_loss_tf or only Q_loss_tf?
        # Ans: No. pi_loss_tf is generic. It is not affected by FW.
        self.Q_loss_tf = self.loss_term(
            batch_tf,
            main_net_fn = partial(main_net_fn, reuse=True),
            target_net_fn = partial(target_net_fn, reuse=True),
            post_process_target_ret = partial(
                tf.clip_by_value,
                clip_value_min=-self.clip_return,
                clip_value_max=(0. if self.clip_pos_returns else np.inf)),
            gamma = self.gamma)

        # NOTE: Why are there separate objective functions for pi and Q?
        # Because the pi loss term makes sure that only pi parameters are in
        # the objective. It selectively optimizes pi parameters.
        # NOTE: vars('main/pi') and vars('main/Q') are exclusive
        # Q_pi_tf = Qₜ(s, μ(s,a))
        self.pi_loss_tf = -tf.reduce_mean(self.main.Q_pi_tf)
        # pi_loss_tf = -Qₜ(s, μ(s,a)) + 1.0*(μ(s, a))²
        self.pi_loss_tf += self.action_l2 * tf.reduce_mean(tf.square(self.main.pi_tf / self.max_u))
        # More loss term and more grads term
        # self.FW_loss_tf = tf.reduce_mean(tf.relu(Q_t(o, ag')+Q_t(o',g) - Q_main(o, g)))
        # FW_grads_tf = tf.gradients(self.FW_loss_tf, self._vars('main/Q'))
        Q_grads_tf = tf.gradients(self.Q_loss_tf, self._vars('main/Q'))
        pi_grads_tf = tf.gradients(self.pi_loss_tf, self._vars('main/pi'))
        assert len(self._vars('main/Q')) == len(Q_grads_tf)
        assert len(self._vars('main/pi')) == len(pi_grads_tf)
        self.Q_grads_vars_tf = zip(Q_grads_tf, self._vars('main/Q'))
        self.pi_grads_vars_tf = zip(pi_grads_tf, self._vars('main/pi'))
        self.Q_grad_tf = flatten_grads(grads=Q_grads_tf, var_list=self._vars('main/Q'))
        self.pi_grad_tf = flatten_grads(grads=pi_grads_tf, var_list=self._vars('main/pi'))
        # TODO: add optional triangular inequality
        # main.Q_tf(o, g) >= target.Q_pi_tf(o, ach_goal(o')) + target.Q_pi_tf(o', g)
        # You can probably create Q_pi_tf from within the batch using o, o_2, g, g_2
        # o -> o_2  g -> g_2?

        # optimizers
        self.Q_adam = MpiAdam(self._vars('main/Q'), scale_grad_by_procs=False)
        self.pi_adam = MpiAdam(self._vars('main/pi'), scale_grad_by_procs=False)

        # polyak averaging
        self.main_vars = self._vars('main/Q') + self._vars('main/pi')
        self.target_vars = self._vars('target/Q') + self._vars('target/pi')
        self.stats_vars = self._global_vars('o_stats') + self._global_vars('g_stats')
        self.init_target_net_op = list(
            map(lambda v: v[0].assign(v[1]), zip(self.target_vars, self.main_vars)))
        self.update_target_net_op = list(
            map(lambda v: v[0].assign(self.polyak * v[0] + (1. - self.polyak) * v[1]), zip(self.target_vars, self.main_vars)))

        # initialize all variables
        tf.variables_initializer(self._global_vars('')).run()
        self._sync_optimizers()
        self._init_target_net()

    def logs(self, prefix=''):
        logs = []
        logs += [('stats_o/mean', np.mean(self.sess.run([self.o_stats.mean])))]
        logs += [('stats_o/std', np.mean(self.sess.run([self.o_stats.std])))]
        logs += [('stats_g/mean', np.mean(self.sess.run([self.g_stats.mean])))]
        logs += [('stats_g/std', np.mean(self.sess.run([self.g_stats.std])))]
        logs += [('train/critic_loss', np.mean(self.log_critic_loss))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def __getstate__(self):
        """Our policies can be loaded from pkl, but after unpickling you cannot continue training.
        """
        excluded_subnames = ['_tf', '_op', '_vars', '_adam', 'buffer', 'sess', '_stats',
                             'main', 'target', 'lock', 'env', 'sample_transitions',
                             'stage_shapes', 'create_actor_critic']

        state = {k: v for k, v in self.__dict__.items() if all([not subname in k for subname in excluded_subnames])}
        state['buffer_size'] = self.buffer_size
        state['tf'] = self.sess.run([x for x in self._global_vars('') if 'buffer' not in x.name])
        return state

    def __setstate__(self, state):
        if 'sample_transitions' not in state:
            # We don't need this for playing the policy.
            state['sample_transitions'] = None

        self.__init__(**state)
        # set up stats (they are overwritten in __init__)
        for k, v in state.items():
            if k[-6:] == '_stats':
                self.__dict__[k] = v
        # load TF variables
        vars = [x for x in self._global_vars('') if 'buffer' not in x.name]
        assert(len(vars) == len(state["tf"]))
        node = [tf.assign(var, val) for var, val in zip(vars, state["tf"])]
        self.sess.run(node)
