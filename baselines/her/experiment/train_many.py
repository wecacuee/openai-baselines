from itertools import product
from functools import partial
from subprocess import check_call
from pathlib import Path
import sys
import json

from .config import DEFAULT_PARAMS
from ..results_plotter import plot_results


class Variations(list):
    pass


def separate_variations(kw):
    return ({k: v for k, v in kw.items()
             if isinstance(v, Variations)},
            {k: v for k, v in kw.items()
             if not isinstance(v, Variations)})


def config_vars_to_configs(config_vars):
    config_keys, config_vals = zip(*config_vars.items())
    return {"-".join(vlist): dict(zip(config_keys, vlist))
            for vlist in product(*config_vals)}


def call_train(**conf):
    cmd = ([sys.executable, str(Path(__file__).parent / "train.py")] +
           sum([["--" + k, str(v)] for k, v in conf.items()], []))
    print("Calling '{}'".format("' '".join(cmd)))
    return check_call(cmd)


def train_many(**kwargs):
    logdirs = []
    variations, kwargs = separate_variations(kwargs)
    for confname, conf in config_vars_to_configs(variations).items():
        conf.update(kwargs)
        call_train(**conf)
        logdirs.append(
            DEFAULT_PARAMS['logdir'](**dict(DEFAULT_PARAMS, **conf)))
    print(logdirs)
    plot_results(logdirs)


train_many_vars = partial(
    train_many,
    env = Variations([
        # "FetchReach-v1",
        # "FetchPush-v1",
        # "FetchSlide-v1",
        # "FetchPickAndPlace-v1",
        # "HandReach-v0",
        "HandManipulateBlock-v0",
        "HandManipulatePen-v0",
        "HandManipulateEgg-v0",
        ]),
    loss_term = Variations(["fwrl", "ddpg", "dqst", "qlst"]),
    replay_strategy = Variations(["future"]))


train_her_fwrl_path_reward = partial(
    train_many,
    env = Variations(["FetchReachPR-v1", "FetchReach-v1"]),
    loss_term = Variations(["dqst", "qlst", "fwrl", "ddpg"]))


train_loss_term_weights = partial(
    train_many,
    env = "FetchSlide-v1",
    loss_term = "fwrl",
    loss_term_weights_json = Variations(
        [json.dumps([i/5, (5-i)/10, (5-i)/10]) for i in range(1,5)]))

main = partial(train_many_vars, num_cpu = 6)
