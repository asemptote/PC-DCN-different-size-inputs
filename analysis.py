import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import brian2.only as br

from contextlib import contextmanager
from functools import partial


def starstar(mapfn, f, kwargs_list):
    """Maps `f` to `kwargs_list` using `mapfn`."""
    return mapfn(partial(apply_kwargs, f), kwargs_list)


def apply_kwargs(f, kwargs):
    """Applies `f` to `kwargs`."""
    return f(**kwargs)


def star(mapfn, f, args_list):
    """Maps `f` to `args_list` using `mapfn`."""
    return mapfn(partial(apply_args, f), args_list)


def apply_args(f, args):
    """Applies `f` to `args`."""
    return f(*args)


def round_lim(array, divisor):
    """Rounds the limits of `array` to the nearest `divisor` and returns the limits."""
    return np.array(
        (
            min(array) - (min(array) % divisor if divisor else 0),
            max(array) - (max(array) % divisor if divisor else 0) + divisor,
        )
    )


def set_lims(xarray, yarray, divisors=(0, 0), ax=None, padding=((0, 0), (0, 0)), minmax=(True,True)):
    """Aligns the limits, major ticks and axis origin."""
    if ax is None:
        ax = plt.gca()
    xlim = round_lim([min(xarray), max(xarray)], divisors[0])
    ylim = round_lim([min(yarray), max(yarray)], divisors[1])
    # print(f'xarray={xarray}, yarray={yarray}, xlim={xlim}, ylim={ylim}')
    ax.set_xlim(xlim + padding[0] * np.diff(xlim))
    ax.set_ylim(ylim + padding[1] * np.diff(ylim))
    ax.spines["left"].set_position(("data", xlim[0]))
    ax.spines["bottom"].set_position(("data", ylim[0]))
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(xlim if minmax[0] else xarray))
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ylim if minmax[1] else yarray))


def echo(expr, echofn=None):
    """Echoes `expr` and returns `expr`."""
    print(echofn(expr) if echofn else expr)
    return expr


@contextmanager
def yieldexpr(expr):
    """Yields `expr` for use in a `with` statement."""
    yield expr


@contextmanager
def seed(seed):
    """Sets the seed for use in a `with` statement."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def fitted_PC_spikes(PC_indices, T, mean_PC_ISI=1 / (80 * br.Hz)):
    """Return PC spikes using Shuting's fit to the experimental ISI distribution."""
    sd_PC_ISI = -0.0015429 * br.second + 0.58358 * mean_PC_ISI  # fit from shuting
    if sd_PC_ISI <= 0 * br.second:
        return ([], [])
    sigma_lognorm = np.sqrt(
        np.log(1 + (sd_PC_ISI**2 / mean_PC_ISI**2))
    )  # dimensionless
    # mu_lognorm is shuting's and wikipedia's convention
    exp_mu_lognorm = mean_PC_ISI**2 / np.sqrt(
        mean_PC_ISI**2 + sd_PC_ISI**2
    )  # has units of mean_PC_ISI
    dist_PC_ISI_unit = br.ms
    dist_PC_ISI = scipy.stats.lognorm(
        s=sigma_lognorm, scale=exp_mu_lognorm / dist_PC_ISI_unit
    )
    return generate_PC_spikes(
        dist_PC_ISI,
        dist_PC_ISI_unit,
        PC_indices,
        T,
        chunk=max(1, int(T / mean_PC_ISI / 10)),
    )


def generate_PC_spikes(ISI_dist, ISI_dist_unit, PC_indices, T, chunk=1):
    """Generates PC spikes using `ISI_dist`."""
    try:
        # if ISI_dist is a list of ISIs to sample from
        iter(ISI_dist)
        samplefn = partial(np.random.choice, ISI_dist)
    except TypeError:
        # elif ISI_dist is a scipy.stats distribution
        samplefn = ISI_dist.rvs
    ISIs = samplefn(size=[len(PC_indices), chunk])
    while min(np.sum(ISIs, axis=1)) * ISI_dist_unit < T:
        ISIs = np.concatenate((ISIs, samplefn(size=[len(PC_indices), chunk])), axis=1)
    PC_times = np.cumsum(ISIs, axis=1).flatten()
    PC_indices = np.full_like(ISIs.T, PC_indices, dtype=int).T.flatten()
    # sort
    argsort = np.argsort(PC_times)
    PC_indices = np.take_along_axis(PC_indices, argsort, axis=0)
    PC_times = np.take_along_axis(PC_times, argsort, axis=0) * ISI_dist_unit
    return PC_indices, PC_times


def xcorr_spike(in_ts, out_ts, window=10 * br.ms):
    """Computes the crosscorrelation between event times in `in_ts` and `out_ts`."""
    in_ts_ms = np.array(in_ts / br.ms)
    out_ts_ms = np.array(out_ts / br.ms)
    window_ms = window / br.ms
    shiftedtimes = (
        np.concatenate(
            [out_t - in_ts_ms[abs(in_ts_ms - out_t) < window_ms] for out_t in out_ts_ms]
            or [[]]
        )
        * br.ms
    )
    return np.histogram(
        shiftedtimes, br.arange(-window + 50 * br.us, window, 100 * br.us)
    )  # np.linspace(-window, window, 101))


def xcorr_tr(in_tr_list, out_tr_list, i, j):
    """Computes the crosscorrelation between `in_tr_list[i]` and `out_tr_list[j]`."""
    return xcorr_spike(in_tr_list[i], out_tr_list[j])
