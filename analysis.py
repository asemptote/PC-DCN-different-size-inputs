import numpy as np
import scipy
import matplotlib.pyplot as plt
import brian2.only as br

from contextlib import contextmanager
from functools import partial

PLOT_DIMS = (2,2)

def starstar(mapfn, f, kwargs_list):
    return mapfn(partial(apply_kwargs, f), kwargs_list)
def apply_kwargs(f, kwargs):
    return f(**kwargs)

def star(mapfn, f, args_list):
    return mapfn(partial(apply_args, f), args_list)
def apply_args(f, args):
    return f(*args)

def subplot(newplot=False, dims=None):
    global plot_counter
    global PLOT_DIMS
    if newplot: plot_counter = 0
    plot_counter += 1
    if dims: PLOT_DIMS = dims
    return plt.subplot(*PLOT_DIMS, plot_counter)

def echo(expr, echofn=None):
    print(echofn(expr) if echofn else expr)
    return expr

@contextmanager
def yieldexpr(expr):
    yield expr
    
@contextmanager
def seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

# def fit_dist(ax, data, dist):
#     x = np.linspace(0, max(data), 100)
#     param = dist.fit(data)#scipy.stats.fit(dist, data)
#     fit = dist.pdf(x, *param)
#     if ax:
#         ax.hist(data, 50, density=True)
#         ax.plot(x, fit, 'r-')
#     return param


def fitted_PC_spikes(PC_indices, T, mean_PC_ISI=1/(80*br.Hz)):
    sd_PC_ISI = -0.0015429*br.second + 0.58358*mean_PC_ISI  # fit from shuting
    if sd_PC_ISI <= 0*br.second: return ([], [])
    sigma_lognorm = np.sqrt(np.log(1 + (sd_PC_ISI**2 / mean_PC_ISI**2))) # dimensionless
    # mu_lognorm is shuting's and wikipedia's convention
    exp_mu_lognorm = mean_PC_ISI**2 / np.sqrt(mean_PC_ISI**2 + sd_PC_ISI**2) # has units of mean_PC_ISI
    dist_PC_ISI_unit = br.ms
    dist_PC_ISI = scipy.stats.lognorm(s=sigma_lognorm, scale=exp_mu_lognorm/dist_PC_ISI_unit)
    return generate_PC_spikes(dist_PC_ISI, dist_PC_ISI_unit, PC_indices, T, chunk=max(1,int(T/mean_PC_ISI/10)))

def generate_PC_spikes(ISI_dist, ISI_dist_unit, PC_indices, T, chunk=1):
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
    # PC_times = np.sort(PC_times) * ISI_dist_unit
    return PC_indices, PC_times

def xcorr_spike(in_ts, out_ts, window=10*br.ms):
    in_ts_ms = np.array(in_ts/br.ms)
    out_ts_ms = np.array(out_ts/br.ms)
    window_ms = window/br.ms
    shiftedtimes = np.concatenate([out_t-in_ts_ms[abs(in_ts_ms-out_t)<window_ms]
                                   for out_t in out_ts_ms] or [[]]) * br.ms
    return np.histogram(shiftedtimes, br.arange(-window+50*br.us, window, 100*br.us))#np.linspace(-window, window, 101))

def xcorr_tr(in_tr_list, out_tr_list, i, j):
    return xcorr_spike(
        in_tr_list[i],
        out_tr_list[j]
        )


def shape():
    # dist = scipy.stats.betaprime
    IPSC = np.genfromtxt('PC_IPSC_20ms.csv')[1:]
    plt.plot(IPSC/sum(IPSC))
    # params = dist.fit(IPSC/sum(IPSC))
    # x = np.linspace(0, 1000, 1000)
    # plt.plot(x, dist.pdf(x, *params))
    # plt.figure()
    EPSC = np.genfromtxt('MF_EPSC_20ms.csv')[1:]
    plt.plot(EPSC/sum(EPSC))
    plt.show()