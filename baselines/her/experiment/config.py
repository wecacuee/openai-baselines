from functools import partial, wraps
import hashlib
import types
import os
import json
import subprocess
from pathlib import Path
from collections import Callable
from inspect import signature, Parameter

import numpy as np
import gym

from baselines.her.pathrewardenv import PathRewardEnv, is_wrapper_instance
from baselines import logger
from baselines.her.ddpg import DDPG, qlearning_loss_term
from baselines.her.her import make_sample_her_transitions
from baselines.her.fwrl import (make_sample_fwrl_transitions,
                                sample_uniform,
                                compute_middle)
from baselines.her.fwrl import (step_with_constraint_loss_term_fwrl,
                                qlearning_constrained_loss_term_fwrl,
                                qlearning_tri_eq_loss_term_fwrl,
                                qlearning_step_loss_term_fwrl,
                                qlearning_step_constrained_loss_term_fwrl,
                                qlearning_step_tri_eq_loss_term_fwrl,
                                step_lower_bound_loss_term_fwrl,
                                step_upper_bound_loss_term_fwrl)


def ignore_extrakw(f):
    """Avoid exceptions like "Got unexpected keyword argument"
    """
    @wraps(f)
    def wrapper(**kw):
        params = signature(f).parameters
        if any((p.kind == Parameter.VAR_KEYWORD) for p in params):
            pass_args = kw
        else:
            need_args = [p.name for p in params]
            pass_args = {k: kw[k] for k in need_args if k in kw}
        return f(**pass_args)
    return wrapper


def git_revision(dir_):
    return subprocess.check_output("git rev-parse --short HEAD".split(),
                                   cwd=dir_).decode("ascii").strip()


this_file_git_rev_fn = ignore_extrakw(
    partial(git_revision,
            Path(__file__).absolute().parent))


class IgnoreFunc(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (float, int, str, bytes, list, dict, tuple)):
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, obj)
        else:
            return ''


def hashkwargs(l=8, **kwargs):
    return hashlib.md5(
        json.dumps(kwargs, cls=IgnoreFunc).encode('utf-8')
    ).hexdigest()[:l]


class GoalNoise:
    @classmethod
    def options(cls):
        return ['uniform', 'zero', 'normal']

    @classmethod
    def from_str(cls, name):
        return getattr(cls, name)

    @staticmethod
    def uniform(goal, distance_threshold=0.05):
        return np.random.rand(*goal.shape) * distance_threshold

    @staticmethod
    def zero(goal, distance_threshold=0.05):
        return 0

    @staticmethod
    def normal(goal, distance_threshold=0.05):
        return np.random.normal(0, distance_threshold / 3, size=goal.shape)


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
        'n_epochs': 60,
    },
    'FetchPush-v1': {
        'n_epochs': 60
    },
    'FetchSlide-v1': {
        'n_epochs': 200
    },
    'FetchPickAndPlace-v1': {
        'n_epochs': 100
    },
    'FetchReachPR-v1': {
        'n_cycles': 10,
        'n_epochs': 100 ,
    },
    'FetchPushPR-v1': {
        'n_epochs': 100
    },
    'FetchSlidePR-v1': {
        'n_epochs': 200
    },
    'FetchPickAndPlacePR-v1': {
        'n_epochs': 200
    },
    'HandReach-v0': {
        'n_epochs': 60
    },
    'HandManipulateBlock-v0': {
        'n_epochs': 200
    },
    'HandManipulateEgg-v0': {
        'n_epochs': 200
    },
    'HandManipulatePen-v0': {
        'n_epochs': 200
    },
    'HandReachPR-v0': {
        'n_epochs': 100
    },
    'HandManipulateBlockPR-v0': {
        'n_epochs': 200
    },
    'HandManipulateEggPR-v0': {
        'n_epochs': 200
    },
    'HandManipulatePenPR-v0': {
        'n_epochs': 200
    },
}


DEFAULT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'baselines.her.actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    'loss_term': 'fwrl',  # Use an additional loss term supported modes: noop or fwrl
    'user': os.environ['USER'],
    'mid_dir': '/z/home/{user}/mid'.format,
    'project_name' : 'floyd-warshall-rl/openai-baselines/her',
    'gitrev': this_file_git_rev_fn,
    'env' : "FetchReach-v1",
    'hash_params' : hashkwargs,
    'env_name' : "FetchReach-v1",
    'logdir': "{mid_dir}/{project_name}/{gitrev}-{exp_name}".format,
    'n_epochs': 50,
    'seed': 0,
    'replay_strategy': 'future',
    'policy_save_interval': 5,
    'clip_return': True,
    'intermediate_sampling': 'uniform', # {uniform|middle}'
    'exp_name': '',
    'recompute_rewards': True,
    'distance_threshold': -1,
    'goal_noise': GoalNoise.zero,
}


CACHED_ENVS = {}


def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def preprocess_params(params):
    new_kw = {}
    for k, v in params.items():
        params[k] = v(**params) if isinstance(v, Callable) else v
        if k.endswith("_json"):
            new_kw[k[:-len("_json")]] = json.loads(v)

    params.update(new_kw)
    return params


def gym_make_kw(env_name=None, distance_threshold=None):
    env = gym.make(env_name)
    if distance_threshold > 0 and hasattr(env.unwrapped, "distance_threshold"):
        env.unwrapped.distance_threshold = distance_threshold
    return env


def prepare_params(kwargs):
    # DDPG params
    ddpg_params = dict()

    env_name = kwargs['env_name']

    make_env = partial(gym_make_kw, env_name=env_name,
                       distance_threshold=kwargs['distance_threshold'])
    kwargs['make_env'] = make_env
    tmp_env = cached_make_env(kwargs['make_env'])
    assert hasattr(tmp_env, '_max_episode_steps')
    kwargs['T'] = tmp_env._max_episode_steps
    assert hasattr(tmp_env.unwrapped, 'distance_threshold')
    kwargs['distance_threshold'] = tmp_env.unwrapped.distance_threshold
    tmp_env.reset()
    kwargs['max_u'] = np.array(kwargs['max_u']) if isinstance(kwargs['max_u'], list) else kwargs['max_u']
    kwargs['gamma'] = 1. - 1. / kwargs['T']
    if 'lr' in kwargs:
        kwargs['pi_lr'] = kwargs['lr']
        kwargs['Q_lr'] = kwargs['lr']
        del kwargs['lr']
    for name in ['buffer_size', 'hidden', 'layers', 'network_class', 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr', 'norm_eps', 'norm_clip',
                 'max_u', 'action_l2', 'clip_obs', 'scope', 'relative_goals',
                 'loss_term']:
        ddpg_params[name] = kwargs[name]
        kwargs['_' + name] = kwargs[name]
        del kwargs[name]
    kwargs['ddpg_params'] = ddpg_params

    env = cached_make_env(kwargs['make_env'])
    if (is_wrapper_instance(env, PathRewardEnv) and
            env.unwrapped.reward_type in PathRewardEnv.MY_REWARD_TYPES):
        kwargs['recompute_rewards'] = False
    else:
        kwargs['recompute_rewards'] = True

    if isinstance(kwargs['goal_noise'], str):
        kwargs['goal_noise'] = partial(
            GoalNoise.from_str(kwargs['goal_noise']),
            distance_threshold=kwargs['distance_threshold'])

    return kwargs


def log_params(params, logger=logger):
    for key in sorted(params.keys()):
        logger.info('{}: {}'.format(key, params[key]))


def get_her_params(params):
    env = cached_make_env(params['make_env'])
    env.reset()

    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)

    # Prepare configuration for HER.
    her_params = {
        'reward_fun': reward_fun,
    }

    for name in ['replay_strategy', 'replay_k', 'recompute_rewards', 'goal_noise']:
        her_params[name] = params[name]
        params['_' + name] = her_params[name]
        del params[name]
    return her_params


def configure_her(params):
    #sample_her_transitions = make_sample_her_transitions(**her_params)
    sample_her_transitions = make_sample_fwrl_transitions(
        **get_her_params(params))

    return sample_her_transitions


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b


available_loss_terms = dict(ddpg=qlearning_loss_term, # 1 term
                            dqst=qlearning_step_loss_term_fwrl, # 2 terms
                            fwrl=qlearning_constrained_loss_term_fwrl, # 3 terms
                            qlst=qlearning_step_constrained_loss_term_fwrl, # 4 terms
                            # Useless below. Do not work even with HER sampling
                            # Experiment 38f4625
                            qste=qlearning_step_tri_eq_loss_term_fwrl,
                            dqte=qlearning_tri_eq_loss_term_fwrl,
                            # Experiment 3f1eafe
                            stfw=step_with_constraint_loss_term_fwrl,
                            stlo=step_lower_bound_loss_term_fwrl,
                            stup=step_upper_bound_loss_term_fwrl)


def loss_term_from_str(
        key,
        available = available_loss_terms):
    return available[key]


available_intermediate_sampling = dict(uniform=sample_uniform,
                                       middle=compute_middle)


def intermediate_sampling_from_str(
        key,
        available = available_intermediate_sampling):
    return available_intermediate_sampling[key]


def get_ddpg_params(dims, params, reuse=False, use_mpi=True, clip_return=True):
    sample_her_transitions = configure_her(params)
    # Extract relevant parameters.
    gamma = params['gamma']
    rollout_batch_size = params['rollout_batch_size']
    ddpg_params = params['ddpg_params']

    input_dims = dims.copy()

    # DDPG agent
    env = cached_make_env(params['make_env'])
    env.reset()
    ddpg_params.update({'input_dims': input_dims,  # agent takes an input observations
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': rollout_batch_size,
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma,
                        })
    ddpg_params['loss_term'] = loss_term_from_str(
        ddpg_params['loss_term'])
    ddpg_params['info'] = {
        'env_name': params['env_name'],
    }
    ddpg_params['reuse'] = reuse
    ddpg_params['use_mpi'] = use_mpi
    return ddpg_params


def configure_ddpg(dims, params, reuse=False, use_mpi=True, clip_return=True):
    ddpg_params = get_ddpg_params(dims, params, reuse=False, use_mpi=True,
                                  clip_return=True)
    policy = DDPG(**ddpg_params)
    return policy


def configure_dims(params):
    env = cached_make_env(params['make_env'])
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())

    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    return dims
