import sys
import numpy as np
from functools import wraps, partial

import gym
import gym.envs.robotics
from gym import Wrapper


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PathRewardEnv(Wrapper):
    SPARSE_PATH = 'sparse-path'
    CONT_STEP_LENGTH = 'cont-step-length'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prev_achieved_goal = None

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        elif self.reward_type == self.SPARSE_PATH:
            return -np.array(1, dtype=np.float32)
        elif self.reward_type == self.CONT_STEP_LENGTH:
            d_ag_pag = goal_distance(self.prev_achieved_goal, achieved_goal)
            return -d_ag_pag
        else:
            return -d

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.prev_achieved_goal = obs['achieved_goal']
        return obs

    def step(self, act):
        obs, rew, done, info = self.env.step(act)
        self.prev_achieved_goal = obs['achieved_goal']
        return obs, rew, done, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)


def apply_wrapper(wrapper, env):
    @wraps(env)
    def env_wrapper(*args, **kwargs):
        return wrapper(env(*args, **kwargs))
    return env_wrapper


def gym_from_env_name(env_name,
                      mapping = dict(
                          HandManipulateBlock="HandBlockEnv",
                          HandManipulateEgg="HandEggEnv",
                          HandManipulatePen="HandPenEnv"),
                      gym_name_template = '{env_name}Env'):
    return mapping.get(env_name, gym_name_template.format(env_name=env_name))


def new_name_and_entry_point(old_env_name,
                             class_name_fn = '{env_name}EnvPR'.format,
                             name_fn = '{env_name}PR-{version}'.format,
                             gym_name_fn = gym_from_env_name):
    parts = old_env_name.split("-")
    env_name = "-".join(parts[:-1])
    version = parts[-1]
    new_class_name = class_name_fn(env_name=env_name)
    new_name = name_fn(env_name=env_name, version=version)
    gym_name = gym_name_fn(env_name=env_name)
    return new_class_name, new_name, apply_wrapper(
        PathRewardEnv, getattr(gym.envs.robotics, gym_name))


def register_wrapped_envs(wrap_envs=("""FetchPush-v1 FetchReach-v1 FetchSlide-v1
                     FetchPickAndPlace-v1
                     HandReach-v0 HandManipulateBlock-v0 HandManipulateEgg-v0
                     HandManipulatePen-v0""").split(),
                          wrap_old_env=new_name_and_entry_point,
                          max_episode_steps=50,
                          kwargs=dict(reward_type=PathRewardEnv.SPARSE_PATH)):
    for old_env_name in wrap_envs:
        new_class_name, new_name, class_ = wrap_old_env(old_env_name)
        # Add the class as an entry point
        setattr(sys.modules[__name__], new_class_name, class_)
        # Register the entry point with gym
        gym.envs.registration.register(
            id=new_name,
            entry_point=":".join((__name__, new_class_name)),
            max_episode_steps=max_episode_steps,
            kwargs=kwargs)

register_wrapped_envs()
register_wrapped_envs(
    wrap_old_env = partial(new_name_and_entry_point,
                           class_name_fn='{env_name}EnvCSL'.format,
                           name_fn = '{env_name}CSL-{version}'.format),
    kwargs=dict(reward_type=PathRewardEnv.CONT_STEP_LENGTH))


def is_wrapper_instance(obj, wrapper_class):
    """
    >>> env = gym.make("FetchReachPR-v1")
    >>> is_wrapper_instance(env, PathRewardEnv)
    True
    >>> env = gym.make("FetchReach-v1")
    >>> is_wrapper_instance(env, PathRewardEnv)
    False
    >>> is_wrapper_instance(env, gym.wrappers.TimeLimit)
    True
    """
    return (isinstance(obj, wrapper_class) or
            (isinstance(obj, Wrapper) and
             is_wrapper_instance(obj.env, wrapper_class)))
