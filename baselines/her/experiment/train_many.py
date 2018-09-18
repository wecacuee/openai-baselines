from itertools import product
from functools import partial
from subprocess import check_call
from pathlib import Path
import sys

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
    plot_results(logdirs)


train_many_vars = partial(
    train_many,
    env = Variations([   "FetchReach-v1",
                         #"FetchPush-v1",
                         #"FetchSlide-v1"
        ]),
    loss_term = Variations([ "stfw", "fwrl", "herr"]),
    replay_strategy = Variations([#"future",
                                  "none"]))


main = partial(train_many_vars, num_cpu = 6)
