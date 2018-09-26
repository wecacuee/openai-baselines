if not __package__ or __package__ == '__main__':
    __package__ = "baselines.her"

from functools import partial, reduce, wraps
from inspect import signature, Parameter

import sys
import os
import os.path as osp
from pathlib import Path
from glob import glob
import json

import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..results_plotter import COLORS


def apply_(arg, f):
    return f(arg)


def compose(*fs):
    return partial(reduce, apply_, fs)


def taketuple(f):
    return wraps(f)(lambda args: f(*args))


def glob_files(dir_, patterns):
    return sum(map(compose(partial(osp.join, dir_), glob), patterns), [])


def progress_load_results(filenames):
    data = []
    for fname in filenames:
        try:
            data.append(pandas.read_csv(fname))
        except pandas.errors.EmptyDataError as e:
            raise RuntimeError("bad file: {}".format(fname)) from e

    return pandas.concat(data)


def common_substr(strs, return_diffs=False):
    """
    >>> common_substr("abcd", "xbcd")
    'bcd'
    """
    strs = list(strs)
    min_len = min(map(len, strs))
    max_len = max(map(len, strs))
    first = strs[0]
    comm = type(first)()
    diffs = [type(first)() for s in strs]
    for i in range(max_len):
        if i < min_len and all((first[i] == s[i]) for s in strs):
            comm.append(first[i])
        else:
            for d, s in zip(diffs, strs):
                if i < len(s):
                    d.append(s[i])
    return (comm, diffs) if return_diffs else comm


def diff_substr(strs, splitre="[-_/]", joinstr="-"):
    import re
    _, diffs = common_substr((re.split(splitre, s) for s in strs),
                             return_diffs=True)
    return map(joinstr.join, diffs)


def mm2inches(mm):
    return mm * 0.03937


def default_figsize(
        A4WIDTH = mm2inches(210)):
    return (A4WIDTH / 2, A4WIDTH / 2 / 1.618)


def jsonloadd(d, paramsjson='params.json'):
    with open(os.path.join(d, paramsjson)) as f:
        params = json.load(f)
    return params


def dict_diffs(params, ignore_keys):
    union_keys = reduce(
        set.intersection, (set(p.keys()) for p in params[1:]),
        set(params[0].keys()))
    union_keys = sorted(union_keys.difference(ignore_keys))
    diffs = [list() for _ in range(len(params))]
    for k in union_keys:
        if not all(params[0].get(k) == p.get(k) for p in params):
            print("diff for key " + k)
            for i, (dif, p) in enumerate(zip(diffs, params)):
                diffs[i].append((k, p.get(k)))
    return diffs


def params_diffs(dirs, jsonloader=jsonloadd,
                 ignore_keys=set(['env_name', 'exp_name', 'logdir',
                                  'hash_params', 'n_epochs'])):
    assert len(dirs) >= 2
    params = list(map(jsonloader, dirs))
    diffs_kv = dict_diffs(params, ignore_keys=ignore_keys)
    return ["-".join(map(str, list(zip(*diff))[1])) for diff in diffs_kv]


def plot_results(
        dirs,
        xdatakey = "epoch",
        metrics = """test/success_rate
        test/mean_Q train/critic_loss test/ag_g_dist""".split(),
        translations={"epoch": "Epoch",
                      "reward_computes": "Reward computes x 1000",
                      "test/success_rate": "Success rate (test)",
                      "test/mean_Q": "Q (test)",
                      "train/critic_loss" : "Critic loss (train)",
                      "test/ag_g_dist": "Distance from goal (test)",
                      "FetchReach-ddpg": "HER (Andrychowicz et al., 2017)",
                      "FetchReach-fwrl": "FWRL (Dhiman et al. 2018)",
                      "FetchReach-dqst": "Ours (goal rewards)",
                      "FetchReachPR-dqst": "Ours (NO goal rewards)",
                      "FetchPush-ddpg": "HER (Andrychowicz et al., 2017)",
                      "FetchPush-fwrl": "FWRL (Dhiman et al. 2018)",
                      "FetchPush-dqst": "Ours (goal rewards)",
                      "FetchPushPR-dqst": "Ours (NO goal rewards)",
                      "FetchSlide-ddpg": "HER (Andrychowicz et al., 2017)",
                      "FetchSlide-fwrl": "FWRL (Dhiman et al. 2018)",
                      "FetchSlide-dqst": "Ours (goal rewards)",
                      "FetchSlidePR-dqst": "Ours (NO goal rewards)",
                      "FetchPickAndPlace-ddpg": "HER (Andrychowicz et al., 2017)",
                      "FetchPickAndPlace-fwrl": "FWRL (Dhiman et al. 2018)",
                      "FetchPickAndPlace-dqst": "Ours (goal rewards)",
                      "FetchPickAndPlacePR-dqst": "Ours (NO goal rewards)",
        },
        pattern = "./progress.csv",
        figsize = default_figsize,
        crop_data = 60):

    f_per_dirs = [glob_files(d, [pattern]) for d in dirs]
    data = {d: add_reward_compute_count(
        progress_load_results(filenames)[:crop_data], d)
            for d, filenames in zip(dirs, f_per_dirs)
            if len(filenames)}
    data_dirs = sorted(data.keys())
    dir_diffs = list(diff_substr(params_diffs(data_dirs)))
    for metric in metrics:
        plt.rc('text', usetex=True)
        fig = plt.figure(figsize=figsize())
        fig.subplots_adjust(left=0.175, bottom=0.20, top=0.98, right=0.98)
        ax = fig.add_subplot(1, 1, 1)
        for d, label, clr in zip(data_dirs, dir_diffs, COLORS):
            if xdatakey in data[d] and metric in data[d]:
                ax.plot(data[d][xdatakey], data[d][metric],
                        label=translations.get(label, label), color=clr)
        ax.set_xlabel(translations.get(xdatakey, xdatakey))
        ax.set_ylabel(translations.get(metric, metric))
        #ax.set_title(translations.get(metric, metric))
        ax.legend(prop=dict(size=6))
        for d in data_dirs:
            path = Path(osp.join(d, xdatakey + "-" + metric + ".pdf"))
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Saving plot to {}".format(path))
            fig.savefig(str(path))


plot_metrics_on_reward_computes = partial(
    plot_results,
    xdatakey = "reward_computes")


def compute_reward_compute_from_epochs(epochs, recompute_rewards,
                                       n_cycles, rollout_batch_size,
                                       batch_size, T=50):
    episodes = epochs * n_cycles * rollout_batch_size
    nsteps = episodes * T

    if recompute_rewards:
        ntrains = epochs * n_cycles
        n_reward_recomputes = ntrains * batch_size
    else:
        n_reward_recomputes = 0
    return (nsteps + n_reward_recomputes) / 1000


def add_reward_compute_count(progress, datadir, jsonloader=jsonloadd):
    params = jsonloader(datadir)
    recompute_rewards = False if "PR-v" in params["env_name"] else True
    reward_computes = compute_reward_compute_from_epochs(
        progress['epoch'], recompute_rewards = recompute_rewards,
        n_cycles = params['n_cycles'],
        rollout_batch_size = params['rollout_batch_size'],
        batch_size = params['batch_size'])
    progress.loc[:, 'reward_computes'] = pandas.Series(
        reward_computes, index=progress.index)
    return progress



def plot_results_grouped(rootdir, dir_patterns, **kw):
    for dirp in dir_patterns:
        plot_results(glob_files(rootdir, patterns = [dirp]), **kw)

def runall(fs, *a, **kw):
    return [f(*a, **kw) for f in fs]


main = partial(
    runall,
    [plot_metrics_on_reward_computes, plot_results])

main_grouped = partial(plot_results_grouped,
                       '/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/',
                       ['*{}-v1-*/'.format(env)
                        for env in "_FetchPush _FetchReach _FetchSlide".split()])


def getdefarg(f, k):
    return signature(f).parameters[k].default


def merged(d1, d2):
    d1c = d1.copy()
    d1c.update(d2)
    return d1c


plot_results_04a8fc6 = partial(
    plot_results,
    dirs = glob("/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/04a8fc6-*-FetchSlide-v1-fwrl-future/"),
    translations = merged(getdefarg(plot_results, 'translations'),
                          {"814a3d24": "[0.6, 0.2, 0.2]",
                           "cc7daced": "[0.2, 0.4, 0.4]",
                           "c26d68c1": "[0.8, 0.1, 0.1]",
                          "923e5525": "[0.4, 0.3, 0.3]"}))

plot_results_path_rewards = partial(
    plot_results,
    dirs = glob("/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/245b3c4-*-FetchReach*-v1-*-future-her_fwrl_path_reward"),
    translations = merged(getdefarg(plot_results, 'translations'),
                          {"FetchReachPR-v1-ddpg": "PR-ddpg",
                             "FetchReach-v1-qlst": "qlst",
                             "FetchReach-v1-ddpg": "ddpg",
                             "FetchReach-v1-dqst": "dqst",
                             "FetchReach-v1-fwrl": "fwrl",
                           "FetchReachPR-v1-qlst": "PR-qlst",
                           "FetchReachPR-v1-fwrl": "PR-fwrl",
                           "FetchReachPR-v1-dqst": "PR-dqst"}))

plot_results_intmdt_sampling = partial(
    plot_results,
    dirs = ["/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/d249d2d-c9bfa98b-FetchPush-v1-fwrl-future", "/z/home/dhiman/mid/floyd-warshall-rl/openai-baselines/her/245b3c4-e5f0dd20-FetchPush-v1-fwrl-future-"],
    translations = merged(getdefarg(plot_results, 'translations'),
                          {"d249d2d-middle": "middle",
                           "245b3c4-uniform": "uniform"}))


if __name__ == '__main__':
    main(sys.argv[1:])
