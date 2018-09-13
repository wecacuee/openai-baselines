if __package__ is None:
    __package__ = "baselines.her.experiment"
from itertools import product

from .config import DEFAULT_PARAMS
from .train import launch
from .results_plotter import plot_rewards


def config_variations(
        keys = ["env_name", "addnl_loss_term"],
        env_name = ["FetchReach-v1", "FetchPush-v0", "FetchSlide-v0"],
        addnl_loss_term = ["fwrl", "noop"]):
    kwargs = locals()
    return {k: kwargs[k] for k in keys}


def config_vars_to_configs(config_vars):
    config_keys, config_vals = zip(*config_vars.items())
    return { "-".join(vlist): dict(zip(config_keys, vlist))
             for vlist in product(*config_vals)}


def config_many(config_vars):
    return {k: conf
            for k, conf in config_vars_to_configs(config_vars).items()}


def train_many(**kwargs):
    logdirs = []
    for confname, conf in config_many(config_variations()).items():
        conf.update(dict(confname = confname))
        conf.update(kwargs)
        params = launch(**dict(DEFAULT_PARAMS, **conf))
        logdirs.append(params['logdir'].format(**params))
    plot_rewards(logdirs)


main = train_many
if __name__ == '__main__':
    main()
