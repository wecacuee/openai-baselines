import sys
import numpy as np
from functools import wraps, partial
from queue import Queue

import gym
import gym.envs.robotics
from gym.envs.registration import registry, load
from gym import Wrapper


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PathRewardEnv(Wrapper):
    SPARSE_PATH = 'sparse-path'
    CONT_STEP_LENGTH = 'cont-step-length'
    MY_REWARD_TYPES = (SPARSE_PATH, CONT_STEP_LENGTH)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.achieved_goal_past_two = [None, None]
        self.reward_compute_count = 0

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        batch_size = goal.shape[0] if goal.ndim >= 2 else 1
        self.reward_compute_count += batch_size
        if self.reward_type == self.SPARSE_PATH:
            if batch_size == 1:
                return np.array(-1.0)
            else:
                return -np.ones(batch_size, dtype=np.float32)
        elif self.reward_type == self.CONT_STEP_LENGTH:
            d_ag_pag = goal_distance(self.achieved_goal_past_two[0], achieved_goal)
            return -d_ag_pag * 100
        else:
            return self.env.compute_reward(achieved_goal, goal, info)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.achieved_goal_past_two[0] = obs['achieved_goal']
        self.achieved_goal_past_two[1] = obs['achieved_goal']
        return obs

    def step(self, act):
        self.achieved_goal_past_two[0] = self.achieved_goal_past_two[1]
        obs, rew, done, info = self.env.step(act)
        new_rew = (self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
                   if (self.reward_type in self.MY_REWARD_TYPES) else rew)
        if self.reward_type not in self.MY_REWARD_TYPES:
            # step calls compute rewards
            self.reward_compute_count += 1
        self.achieved_goal_past_two[1] = obs['achieved_goal']
        info['reward_compute_count'] = self.reward_compute_count
        return obs, new_rew, done, info

    def __getattr__(self, attr):
        return getattr(self.env, attr)


def apply_wrapper(wrapper, env):
    @wraps(env)
    def env_wrapper(*args, **kwargs):
        return wrapper(env(*args, **kwargs))
    return env_wrapper


def gym_spec_from_env_id(env_id):
    return registry.env_specs[env_id]


def new_name_and_entry_point(old_env_name,
                             class_name_fn = '{env_name}EnvPR'.format,
                             name_fn = '{env_name}PR-{version}'.format,
                             gym_spec_fn = gym_spec_from_env_id):
    parts = old_env_name.split("-")
    env_name = "-".join(parts[:-1])
    version = parts[-1]
    new_class_name = class_name_fn(env_name=env_name)
    new_name = name_fn(env_name=env_name, version=version)
    gym_spec = gym_spec_fn(old_env_name)
    gym_entry_point = load(gym_spec._entry_point)
    return new_class_name, new_name, apply_wrapper(
        PathRewardEnv, gym_entry_point), gym_spec


def register_wrapped_envs(wrap_envs=("""FetchPush-v1 FetchReach-v1 FetchSlide-v1
                                        FetchPickAndPlace-v1
                                        HandReach-v0 HandManipulateBlock-v0
                                        HandManipulateEgg-v0
                                        HandManipulatePen-v0
                                        HandManipulateBlockRotateXYZ-v0
                                        HandManipulatePenRotate-v0
                                        HandManipulateEggFull-v0
                                """).split(),
                          wrap_old_env=new_name_and_entry_point,
                          max_episode_steps=50,
                          kwargs=dict(reward_type=PathRewardEnv.SPARSE_PATH)):
    for old_env_name in wrap_envs:
        new_class_name, new_name, class_, gym_spec = wrap_old_env(old_env_name)
        # Add the class as an entry point
        setattr(sys.modules[__name__], new_class_name, class_)
        new_kwargs = gym_spec._kwargs.copy()
        new_kwargs.update(kwargs)
        # Register the entry point with gym
        gym.envs.registration.register(
            id=new_name,
            entry_point=":".join((__name__, new_class_name)),
            trials=gym_spec.trials,
            reward_threshold=gym_spec.reward_threshold,
            local_only=gym_spec._local_only,
            kwargs=new_kwargs,
            nondeterministic=gym_spec.nondeterministic,
            tags=gym_spec.tags,
            max_episode_steps=gym_spec.max_episode_steps,
            max_episode_seconds=gym_spec.max_episode_seconds,
            timestep_limit=gym_spec.timestep_limit,
        )

register_wrapped_envs()
register_wrapped_envs(
    wrap_old_env = partial(new_name_and_entry_point,
                           class_name_fn='{env_name}EnvCSL'.format,
                           name_fn = '{env_name}CSL-{version}'.format),
    kwargs=dict(reward_type=PathRewardEnv.CONT_STEP_LENGTH))
register_wrapped_envs(
    wrap_old_env = partial(new_name_and_entry_point,
                           class_name_fn='{env_name}EnvSparse'.format,
                           name_fn = '{env_name}Sparse-{version}'.format),
    kwargs=dict(reward_type='sparse'))


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

def dummy():
    """
    >>> import gym
    >>> env = gym.make("FetchReachCSL-v1")
    >>> _ = env.seed(0)
    >>> _ = env.reset()
    >>> for _ in range(5):
    ...     obs, rew, _, info = env.step(env.action_space.sample())
    ...     rew2 = env.compute_reward(obs['achieved_goal'], obs['desired_goal'], dict())
    ...     print(rew, rew2, info['reward_compute_count'])
    -1.4871227185677292 -1.4871227185677292 1
    -1.1468296518239358 -1.1468296518239358 3
    -3.2561153606471986 -3.2561153606471986 5
    -3.6017700571537192 -3.6017700571537192 7
    -3.9249514565213195 -3.9249514565213195 9

    >>> env = gym.make("FetchReachPR-v1")
    >>> _ = env.seed(0)
    >>> _ = env.reset()
    >>> for _ in range(5):
    ...     obs, rew, _, info = env.step(env.action_space.sample())
    ...     rew2 = env.compute_reward(obs['achieved_goal'], obs['desired_goal'], dict())
    ...     print(rew, rew2, info['reward_compute_count'])
    -1.0 -1.0 1
    -1.0 -1.0 3
    -1.0 -1.0 5
    -1.0 -1.0 7
    -1.0 -1.0 9


    >>> env = gym.make("FetchReachSparse-v1")
    >>> _ = env.seed(0)
    >>> _ = env.reset()
    >>> for _ in range(5):
    ...     obs, rew, _, info = env.step(env.action_space.sample())
    ...     rew2 = env.compute_reward(obs['achieved_goal'], obs['desired_goal'], dict())
    ...     print(rew, rew2, info['reward_compute_count'])
    -1.0 -1.0 1
    -1.0 -1.0 3
    -1.0 -1.0 5
    -1.0 -1.0 7
    -1.0 -1.0 9
    """
    pass
