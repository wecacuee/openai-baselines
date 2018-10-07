if not __package__ or __package__ == '__main__':
    __package__ = "baselines.her"

from functools import partial, reduce, wraps
from inspect import signature, Parameter

import re
import sys
import os
import os.path as osp
from pathlib import Path
from glob import glob
import json
import numpy as np

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
    comm, diffs = common_substr((re.split(splitre, s) for s in strs),
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
        params.setdefault("distance_threshold", 0.05)
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


def moving_average(a, n=5):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    ma = ret / n
    ma[:n] = ret[n-1]
    return ma


class Xargs:
    def __init__(self, func, expect_args=()):
        self.func = func
        self.expect_args = expect_args


class EvalDict:
    def __init__(self, dct):
        self.dct = dct

    def __getitem__(self, key):
        v = self.dct[key]
        return (v.func(**{k: self[k] for k in v.expect_args})
                if isinstance(v, Xargs) else v)


def default_kw(func):
    return {k: p.default for k, p in signature(func).parameters.items()
            if p.default is not Parameter.empty}


def need_args(func):
    return [k for k, p in signature(func).parameters.items()
            if p.kind == Parameter.POSITIONAL_OR_KEYWORD]


def eval_kwargs(func, evaldictclass=EvalDict):
    def_kw = default_kw(func)
    need_a = need_args(func)

    @wraps(func)
    def wrapper(*a, **kw):
        a2kw = dict(zip(need_a[:len(a)], a))
        kwc = def_kw.copy()
        kwc.update(a2kw)
        kwc.update(kw)
        ekw = evaldictclass(kwc)
        kwevaled = {k: ekw[k] for k in need_a}
        return func(**kwevaled)

    return wrapper


def mplib_default_figure(
        subplots_adjust = dict(left=0.175, bottom=0.20, top=0.98, right=0.98),
        figsize_gen=default_figsize, **kwargs):
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=figsize_gen(), **kwargs)
    fig.subplots_adjust(**subplots_adjust)
    return fig


def mplib_default_axes(fig, subplots=(1, 1, 1)):
    return fig.add_subplot(*subplots)


@eval_kwargs
def mplib_plot(line_plots,
               xlabel,
               ylabel,
               fpath,
               legend_prop = dict(prop=dict(size=12)),
               xlabel_props = dict(fontsize=12),
               ylabel_probs = dict(fontsize=12),
               fig = Xargs(mplib_default_figure),
               ax = Xargs(mplib_default_axes, ("fig",))):
    for x, y, kw in line_plots:
        ax.plot(x, y, **kw)
    ax.set_xlabel(xlabel, **xlabel_props)
    ax.set_ylabel(ylabel, **ylabel_probs)
    ax.legend(**legend_prop)
    ax.figure.savefig(fpath + ".pdf")
    plt.close(ax.figure)


@eval_kwargs
def plotly_plot(line_plots,
                xlabel,
                ylabel,
                fpath):
    import plotly.offline as plt
    import plotly.graph_objs as go
    data = [go.Scatter(x = x, y = y, name=kw['label'])
            for x, y, kw in line_plots]
    plt.plot(dict(data=data,
                  layout=go.Layout(xaxis=dict(title=xlabel),
                                   yaxis=dict(title=ylabel))),
             filename=(fpath + ".html"), auto_open=False, show_link=False)


def plot_results(
        dirs,
        xdatakey = "epoch",
        metrics = """test/success_rate
        test/mean_Q train/critic_loss test/ag_g_dist""".split(),
        translations={"epoch": "Epoch",
                      "reward_computes": "Reward computes / 1000",
                      "test/success_rate": "Success rate (test)",
                      "test/mean_Q": "Q (test)",
                      "train/critic_loss" : "Critic loss (train)",
                      "test/ag_g_dist": "Distance from goal (test)",
                      "FetchReach-ddpg": "HER",
                      "FetchReach-fwrl": "FWRL",
                      "FetchReach-dqst": "Ours (with goal rewards)",
                      "FetchReachPR-dqst": "Ours",
                      "FetchPush-ddpg": "HER",
                      "FetchPush-fwrl": "FWRL",
                      "FetchPush-dqst": "Ours (with goal rewards)",
                      "FetchPushPR-dqst": "Ours",
                      "FetchSlide-ddpg": "HER",
                      "FetchSlide-fwrl": "FWRL",
                      "FetchSlide-dqst": "Ours (with goal rewards)",
                      "FetchSlidePR-dqst": "Ours",
                      "FetchPickAndPlace-ddpg": "HER",
                      "FetchPickAndPlace-fwrl": "FWRL",
                      "FetchPickAndPlace-dqst": "Ours (with goal rewards)",
                      "FetchPickAndPlacePR-dqst": "Ours",
                      "HandReach-ddpg": "HER",
                      "HandReach-fwrl": "FWRL",
                      "HandReach-dqst": "Ours (with goal rewards)",
                      "HandReachPR-dqst": "Ours",
                      "HandManipulateBlockRotateXYZ-ddpg": "HER",
                      "HandManipulateBlockRotateXYZ-fwrl": "FWRL",
                      "HandManipulateBlockRotateXYZ-dqst": "Ours (with goal rewards)",
                      "HandManipulateBlockRotateXYZPR-dqst": "Ours",
                      "HandManipulateEggFull-ddpg": "HER",
                      "HandManipulateEggFull-fwrl": "FWRL",
                      "HandManipulateEggFull-dqst": "Ours (with goal rewards)",
                      "HandManipulateEggFullPR-dqst": "Ours",
                      "HandManipulatePenRotate-ddpg": "HER",
                      "HandManipulatePenRotate-fwrl": "FWRL",
                      "HandManipulatePenRotate-dqst": "Ours (with goal rewards)",
                      "HandManipulatePenRotatePR-dqst": "Ours",
                      "HandManipulatePenRotate-zero-ddpg": "HER",
                      "HandManipulatePenRotate-zero-fwrl": "FWRL",
                      "HandManipulatePenRotatePR-uniform-dqst": "Ours",
                      "FetchPushPR-605e7e1-ddpg": "Ours (No step loss)",
                      "FetchPush-6efc1de-ddpg": "HER",
                      "FetchPushPR-6efc1de-dqst": "Ours",
                      "FetchPickAndPlacePR-605e7e1-ddpg": "Ours (No step loss)",
                      "FetchPickAndPlace-6efc1de-ddpg": "HER",
                      "FetchPickAndPlacePR-6efc1de-dqst": "Ours",
                      "0.01-FetchPush-6efc1de-ddpg": "HER, $\epsilon = 0.01$",
                      "0.05-FetchPush-be0910c-ddpg": "HER, $\epsilon = 0.05$",
                      "0.001-FetchPush-be467df-ddpg": "HER, $\epsilon = 0.001$",
                      "0.01-6efc1de": "HER, $\epsilon = 0.01$",
                      "0.05-be0910c": "HER, $\epsilon = 0.05$",
                      "0.001-be467df": "HER, $\epsilon = 0.001$",
                      "0.01-FetchPush-6efc1de": "Ours, $\epsilon = 0.01$",
                      "0.05-FetchPushPR-be0910c": "Ours, $\epsilon = 0.05$",
                      "0.001-FetchPushPR-be467df": "Ours, $\epsilon = 0.001$",
        },
        pattern = "./progress.csv",
        moving_average_n = partial(moving_average, n = 5),
        #savdir = "/tmp/ablate-ddpg-dqst-low_tresh_chosen-low_thresh_alt-dqst",
        #savdir = "/tmp/ablate-ddpg-with-without-step-loss/",
        savdir = None,
        crop_data = 30,
        plot_lines = plotly_plot
):

    f_per_dirs = [glob_files(d, [pattern]) for d in dirs]
    data = {d: add_reward_compute_count(
        progress_load_results(filenames)[:crop_data], d)
            for d, filenames in zip(dirs, f_per_dirs)
            if len(filenames)}
    data_dirs = sorted(data.keys())
    dir_diffs = list(diff_substr(params_diffs(data_dirs)))
    uniq_diff_str = "-".join(sorted(set(sum(
        [re.split("[-_/]", dif) for dif in dir_diffs], []))))
    for metric in metrics:
        line_plots = []
        for d, label, clr in zip(data_dirs, dir_diffs, COLORS):
            if xdatakey in data[d] and metric in data[d]:
                line_plots.append(
                    (data[d][xdatakey],
                     data[d][metric].values,
                     dict(label=translations.get(label, label), color=clr)))

        savefig = partial(plot_lines,
                          line_plots=line_plots,
                          xlabel=translations.get(xdatakey, xdatakey),
                          ylabel=translations.get(metric, metric))
        if not savdir:
            path = Path(osp.join(
                osp.dirname(dirs[0]),
                uniq_diff_str, xdatakey + "-" + metric))
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Saving plot to {}".format(path))
            savefig(fpath=str(path))
        else:
            path = Path(osp.join(savdir, label + xdatakey + "-" + metric))
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Saving plot to {}".format(path))
            savefig(fpath=str(path))


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
