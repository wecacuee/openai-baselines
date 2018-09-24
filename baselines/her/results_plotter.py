if __package__ is None:
    __package__ = "baselines.her"

from functools import partial, reduce, wraps
from inspect import signature, Parameter

import sys
import os
import os.path as osp
from pathlib import Path
from glob import glob
import json

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
    import pandas
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


def diff_substr(strs, s, splitre="[-_/]", joinstr="-"):
    import re
    _, diffs = common_substr((re.split(splitre, s) for s in strs),
                             return_diffs=True)
    return joinstr.join(diffs[strs.index(s)])


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
                 ignore_keys=set(['env_name', 'logdir', 'hash_params'])):
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
                      "test/success_rate": "Success rate (test)",
                      "test/mean_Q": "Q (test)",
                      "train/critic_loss" : "Critic loss (train)",
                      "test/ag_g_dist": "Distance from goal (test)",
        },
        pattern = "./progress.csv",
        figsize = default_figsize):

    f_per_dirs = [glob_files(d, [pattern]) for d in dirs]
    dir_diffs = params_diffs(dirs)
    data = {d: progress_load_results(filenames)
            for d, filenames in zip(dirs, f_per_dirs)
            if len(filenames)}
    for metric in metrics:
        fig = plt.figure(figsize=figsize())
        fig.subplots_adjust(left=0.175, bottom=0.20, top=0.98, right=0.98)
        ax = fig.add_subplot(1, 1, 1)
        for d, label, clr in zip(data.keys(), dir_diffs, COLORS):
            if xdatakey in data[d] and metric in data[d]:
                ax.plot(data[d][xdatakey], data[d][metric],
                        label=translations.get(label, label), color=clr)
        ax.set_xlabel(translations.get(xdatakey, xdatakey))
        ax.set_ylabel(translations.get(metric, metric))
        #ax.set_title(translations.get(metric, metric))
        ax.legend()
        for d in data.keys():
            path = Path(osp.join(d, metric + ".pdf"))
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Saving plot to {}".format(path))
            fig.savefig(str(path))

        if os.environ.get("DISPLAY") == ":0":
            pass
            #plt.show()


def plot_results_grouped(rootdir, dir_patterns, **kw):
    for dirp in dir_patterns:
        plot_results(glob_files(rootdir, patterns = [dirp]), **kw)

main = plot_results

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


if __name__ == '__main__':
    main(sys.argv[1:])
