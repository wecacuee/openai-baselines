if __package__ is None:
    __package__ = "baselines.her"

from functools import partial, reduce, wraps

import sys
import os
import os.path as osp
from pathlib import Path
from glob import glob

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


def diff_substr(strs, s):
    _, diffs = common_substr(map(list, strs), return_diffs=True)
    return "".join(diffs[strs.index(s)])


def mm2inches(mm):
    return mm * 0.03937


def default_figsize(
        A4WIDTH = mm2inches(210)):
    return (A4WIDTH / 2, A4WIDTH / 2 / 1.618)


def plot_results(
        dirs,
        xdatakey = "epoch",
        metrics = """test/success_rate
        test/mean_Q train/critic_loss train/critic_addnl_loss""".split(),
        translations={"epoch": "Epoch",
                      "test/success_rate": "Success rate (test)",
                      "test/mean_Q": "Q (test)",
                      "train/critic_loss" : "Critic loss (train)",
                      "train/critic_addnl_loss" : "FWRL Critic loss (train)",
        },
        pattern = "./progress.csv",
        figsize = default_figsize):

    f_per_dirs = [glob_files(d, [pattern]) for d in dirs]
    data = {d: progress_load_results(filenames)
            for d, filenames in zip(dirs, f_per_dirs)
            if len(filenames)}
    for metric in metrics:
        fig = plt.figure(figsize=figsize())
        fig.subplots_adjust(left=0.175, bottom=0.20, top=0.98, right=0.98)
        ax = fig.add_subplot(1, 1, 1)
        for d, clr in zip(data.keys(), COLORS):
            if xdatakey in data[d] and metric in data[d]:
                label = diff_substr(dirs, d)
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

if __name__ == '__main__':
    main(sys.argv[1:])
