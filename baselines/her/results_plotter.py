if __package__ is None:
    __package__ = "baselines.her"

from functools import partial, reduce, wraps

import sys
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


def progress_load_results(dir_, pattern="progress.csv"):
    import pandas
    data = []
    for fname in glob_files(dir_, patterns = [pattern]):
        data.append(pandas.read_csv(fname))
    return pandas.concat(data)


def common_substr(strs, return_diffs=False):
    """
    >>> common_substr("abcd", "xbcd")
    'bcd'
    """
    strs = list(strs)
    min_len = min(map(len, strs))
    first = strs[0]
    comm = type(first)()
    diffs = [type(first)() for s in strs]
    for i in range(min_len):
        if all((first[i] == s[i]) for s in strs):
            comm.append(first[i])
        else:
            for d, s in zip(diffs, strs):
                d.append(s[i])
    return (comm, diffs) if return_diffs else comm


def diff_substr(strs, s):
    _, diffs = common_substr(map(list, strs), return_diffs=True)
    return "".join(diffs[strs.index(s)])


def plot_results(
        dirs,
        xdatakey = "epoch",
        metrics = """train/success_rate test/success_rate
        test/mean_Q stats_o/mean stats_g/mean""".split()):
    data = {d : progress_load_results(d) for d in dirs}
    for metric in metrics:
        plt.clf()
        for d, clr in zip(dirs, COLORS):
            label = diff_substr(dirs, d)
            plt.plot(data[d][xdatakey], data[d][metric],
                     label=label, color=clr)
        plt.xlabel(xdatakey)
        plt.ylabel(metric)
        plt.title(metric)
        plt.legend()
        plt.show()
        path = Path(osp.join(d, metric + ".pdf"))
        path.parent.mkdir(parents=True, exist_ok=True)
        print("Saving plot to {}".format(path))
        plt.savefig(str(path))


main = plot_results

if __name__ == '__main__':
    main(sys.argv[1:])
