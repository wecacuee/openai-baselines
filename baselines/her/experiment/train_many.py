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
    return {"-".join(map(str, vlist)): dict(zip(config_keys, vlist))
            for vlist in product(*config_vals)}


def call_train(**conf):
    cmd = ([sys.executable, str(Path(__file__).parent / "train.py")] +
           sum([["--" + k, str(v)] for k, v in conf.items()], []))
    print("Calling '{}'".format("' '".join(cmd)))
    return check_call(cmd)


def run_one_experiment(confname, var_conf, common_kwargs):
    var_conf.update(common_kwargs)
    var_conf['exp_name'] = "-".join((var_conf.get('exp_name', ''), confname))
    call_train(**var_conf)


def experiment_configurations(kwargs):
    variations, common_kwargs = separate_variations(kwargs)
    return config_vars_to_configs(variations), common_kwargs


def train_many(**kwargs):
    logdirs = []
    var_configs, common_kwargs = experiment_configurations(kwargs)
    for confname, conf in var_configs.items():
        run_one_experiment(confname, conf, common_kwargs)
        logdirs.append(
            DEFAULT_PARAMS['logdir'](**dict(DEFAULT_PARAMS, **conf)))
    print(logdirs)
    plot_results(logdirs)

def environ_list():
    return [
        # "FetchReachSparse-v1",
        # "FetchPushSparse-v1",
        # "FetchSlideSparse-v1",
        "FetchPickAndPlaceSparse-v1",
        "HandReachSparse-v0",
        "HandManipulateBlockSparse-v0",
        "HandManipulatePenSparse-v0",
        "HandManipulateEggSparse-v0",
        ]

train_many_vars = partial(
    train_many,
    exp_name = 'many_envs',
    env = Variations([
        # "FetchReachSparse-v1",
        # "FetchPushSparse-v1",
        # "FetchSlideSparse-v1",
        # "FetchPickAndPlaceSparse-v1",
        # "HandReachSparse-v0",
        "HandManipulateBlockSparse-v0",
        "HandManipulatePenSparse-v0",
        "HandManipulateEggSparse-v0",
        ]),
    loss_term = Variations(["fwrl", "ddpg", "dqst", "qlst"]),
    replay_strategy = Variations(["future"]))


train_her_fwrl_path_reward = partial(
    train_many,
    exp_name = 'path_reward',
    n_epochs = 60,
    env = Variations(["FetchSlidePR-v1", "FetchSlideSparse-v1"]),
    loss_term = Variations(["dqst", "qlst", "fwrl", "ddpg"]))


exp_conf_path_reward = partial(
    experiment_configurations,
    exp_name = 'path_reward',
    n_epochs = 60,
    env = Variations([
        "FetchPickAndPlaceSparse-v1",
        "FetchPickAndPlacePR-v1",
        "HandReachSparse-v0",
        "HandReachPR-v0",
        "HandManipulateBlockSparse-v0",
        "HandManipulateBlockPR-v0",
        "HandManipulatePenSparse-v0",
        "HandManipulatePenPR-v0",
        "HandManipulateEggSparse-v0",
        "HandManipulateEggPR-v0"],
    ),
    loss_term = Variations(["dqst", "qlst", "fwrl", "ddpg"]))


train_loss_term_weights = partial(
    train_many,
    exp_name = 'loss_term_weights',
    n_epochs = 20,
    env = "FetchPushSparse-v1",
    loss_term = "qlst",
    loss_term_weights_json = Variations(
        [json.dumps([i/max(1,i+j),
                     1-(i+j)/max(1,i+j),
                     j/max(1,i+j),
                     j/max(1,i+j)])
         for i, j in product(range(3), repeat=2)]))


train_intmdt_sampling = partial(
    train_many,
    exp_name = 'intmdt_sampling',
    n_epochs = 20,
    env = "FetchPushSparse-v1",
    loss_term = "fwrl",
    intermediate_sampling = Variations([
        'middle', 'uniform']))


main = partial(train_many_vars, num_cpu = 6)
