from itertools import product
from functools import partial
from subprocess import check_call
from pathlib import Path
import sys

from .config import DEFAULT_PARAMS
from ..results_plotter import plot_results


def config_variations(
        keys = ["env", "addnl_loss_term"],
        env = [  # "FetchReach-v1",
            #"FetchPush-v1",
            "FetchSlide-v1"],
        addnl_loss_term = [ "stepfwrl", "fwrl", "noop"],
        replay_strategy = ["future", "none"]):
    kwargs = locals()
    return {k: kwargs[k] for k in keys}


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
    for confname, conf in config_vars_to_configs(config_variations()).items():
        conf.update(kwargs)
        call_train(**conf)
        logdirs.append(
            DEFAULT_PARAMS['logdir'].format(**dict(DEFAULT_PARAMS, **conf)))
    plot_results(logdirs)


main = partial(train_many, num_cpu = 6)
