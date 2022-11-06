import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

import brian2 as br

#import functions

plot_dims = (6,6)
def subplot(newplot=False):
    global plot_counter
    if newplot: plot_counter = 0
    plot_counter += 1
    return plt.subplot(*plot_dims, plot_counter)

def execfn(*args):
    exec(' '.join(args))
    
def echo(expr):
    print(expr)
    return expr

from functools import partial

def starstarmap(p, f, kwargs_list):
    return p.map(partial(apply_kwargs, f), kwargs_list)

def apply_kwargs(f, kwargs):
    return f(**kwargs)


def fig2_fitted(**kwargs):
    '''
    Use as e.g.
        python PC_DCN_brian.py execfn fig2_fitted(T=10*br.second, N_DCN=10, N_extE_DCN=30000, fix_gI_ylim=False)
        python PC_DCN_brian.py execfn fig2_fitted(T=10*br.second, N_DCN=10, N_extE_DCN=30000, fix_gI_ylim=False, dist=scipy.stats.pareto)
    '''
    import functions
    G = np.genfromtxt('G.csv')  # nS
    
    plt.figure(figsize=(10,10))
    global plot_dims
    plot_dims = (6,6)
    ax = subplot(newplot=True)
    
    dist = kwargs.get('dist', scipy.stats.lognorm)
    
    param = functions.plot_fit(ax, G, dist)
    plt.title(dist.name)
    plt.xlabel('size (nS)')
    
    kwargs['input_sizes'] = np.sort(dist(*param).rvs(30))*br.nS
    kwargs['neuronpartition'] = range(29)
    kwargs['newfig'] = False
    if 'N_extE_DCN' not in kwargs: kwargs['N_extE_DCN'] = 30000
    
    ax = subplot()
    fig2(**kwargs)

def fig2(**kwargs):
    '''
    pass in all the input sizes and measure the cross correlation for each
    spike size
    '''
    input_sizes = kwargs.get('input_sizes', 
                             np.concatenate([np.repeat(3, 16),
                                             np.repeat(10, 10),
                                             np.repeat(30, 2)])*br.nS)
    N_inputs = len(input_sizes)
    if 'N_DCN' not in kwargs: kwargs['N_DCN'] = 1
    kwargs['N_PC'] = N_inputs * kwargs['N_DCN']
    kwargs['connect'] = {'i': [], 'j': []}
    kwargs['w'] = []
    for j_DCN in range(kwargs['N_DCN']):
        kwargs['connect']['i'].extend(range(N_inputs*j_DCN, N_inputs*(j_DCN+1)))
        kwargs['connect']['j'].extend([j_DCN for _ in input_sizes])
        kwargs['w'].extend(input_sizes)
    
    # kwargs['connect'] = {'i': range(len(input_sizes)), 'j': 0}
    # kwargs['w'] = input_sizes
    # kwargs['N_DCN'] = 1
    # kwargs['N_PC'] = len(input_sizes)
    if 'N_extE_DCN' not in kwargs: kwargs['N_extE_DCN'] = 5000
    
    net = run_DCN_conductance(**kwargs)
    
    timefilter = net.t - net['st_DCN'].t < 200*br.ms
    
    neuronpartition = kwargs.get('neuronpartition', [0,16,26])
    n_subplots = len(neuronpartition) + 2
    
    if kwargs.get('newfig', True):
        plt.figure(figsize=(10,10))
        global plot_dims
        plot_dims = (int(np.ceil(n_subplots**0.5)),int(np.floor(n_subplots**0.5)))
        ax = subplot(newplot=True)
    
    DCN_neuronindex = 0
    
    # plot the total inhibitory conductance
    plt.plot(net['st_DCN'].t[timefilter]/br.second, net['st_DCN'].gI[DCN_neuronindex][timefilter]/br.nS)
    plt.xlabel('time (s)')
    plt.ylabel('I conductance (nS)')
    #plt.title(f'{N_inputs_list[neuronindex]} x {input_sizes[neuronindex]/br.nS}nS')
    if kwargs.get('fix_gI_ylim', True): plt.ylim([0, 100])
    
    ax = subplot()   
    
    # plot the V trace of the DCN neuron
    plt.plot(net['st_DCN'].t[timefilter]/br.second, net['st_DCN'].v[DCN_neuronindex][timefilter]/br.mV)
    for t in net['sp_DCN'].spike_trains()[DCN_neuronindex]:
        if net.t - t < 200*br.ms: plt.axvline(t/br.second, 0, 100, c='black')
    plt.title(f"mean DCN rate = {len(net['sp_DCN'].spike_trains()[DCN_neuronindex])/net.t}")
    plt.xlabel('time (s)')
    plt.ylabel('V (mV)')
    plt.ylim([-75, 0])
    
    
    
    for i in range(len(neuronpartition)):
        ax = subplot()
        # plot the PC spike triggered average DCN firing rate 
        #hist, bins = STA_rate(ax, net, neuronindex)
        hist_list, bins_list = zip(*[spike_crosscorrelogram(
            np.concatenate([net['sp_PC'].spike_trains()[PC_neuronindex]
                            for PC_neuronindex in range(N_inputs*j_DCN+neuronpartition[i],
                                                        N_inputs*j_DCN+(neuronpartition[i+1] if i<=len(neuronpartition)-2 else N_inputs))] or [[]]),
            net['sp_DCN'].spike_trains()[j_DCN]
            ) for j_DCN in range(kwargs['N_DCN'])])
        hist = np.sum(hist_list, axis=0)
        bins = bins_list[0]
        ax.stairs(hist/(hist[0] or 1), bins/br.ms)
        #plt.title(f"N_PC = {len(net['PC'])} \n T = {net.t}")
        plt.title(f'{input_sizes[neuronpartition[i]]/br.nS:.3g} nS')
        plt.ylabel('spikes (norm.)')
        plt.xlabel('t (ms)')
        plt.ylim([0,2])
    
    if kwargs.get('net', False): return net
    else: plt.show()


def fig4(**kwargs):
    '''
    choose the numbers of inputs to pass in given the total conductance
    '''
    
    totalconductance = 200*br.nS
    
    if 'N_PC' not in kwargs: kwargs['N_PC'] = 80
    if 'N_extE_DCN' not in kwargs: kwargs['N_extE_DCN'] = 5000
    
    connect = {'i': [], 'j': []}
    w = []
    input_sizes = []
    N_inputs_list = kwargs.get('N_inputs_list',
                               list(range(1, kwargs['N_PC']+1, kwargs.get('step', 1)))+[False])
    for j_DCN, N_inputs in enumerate(N_inputs_list):
        w_curr = totalconductance / N_inputs if N_inputs else 0*br.nS
        connect['i'].extend(range(N_inputs))
        connect['j'].extend([j_DCN for _ in range(N_inputs)])
        w.extend([w_curr for _ in range(N_inputs)])
        input_sizes.append(w_curr)
    kwargs['N_DCN'] = j_DCN + 1
    kwargs['connect'] = connect
    kwargs['w'] = w
    kwargs['tauI'] = [2.9*br.ms for _ in range(kwargs['N_DCN']-1)] + [np.inf*br.ms]
    if 'gI' not in kwargs: kwargs['gI'] = 45*br.nS
    
    net = run_DCN_conductance(**kwargs)
    
    neuronindices = kwargs.get('neuronindices', [N_inputs_list.index(N_inputs) for N_inputs in [0, 80, 40, 20, 10, 5]])
    STA_rates = {DCN_neuronindex: spike_crosscorrelogram(
        np.concatenate([net['sp_PC'].spike_trains()[PC_neuronindex]
                        for PC_neuronindex in np.array(net['S'].i)[np.array(net['S'].j)==DCN_neuronindex]] or [[]]),
        net['sp_DCN'].spike_trains()[DCN_neuronindex])
        for DCN_neuronindex in neuronindices}
    
    lastsecondfilter = net.t - net['st_DCN'].t < br.second
    
    plt.figure(figsize=(10,10))
    global plot_dims
    plot_dims = (7,4)
    ax = subplot(newplot=True)
    
    for neuronindex in neuronindices:
        # plot the total inhibitory conductance
        plt.plot(net['st_DCN'].t[lastsecondfilter]/br.second, net['st_DCN'].gI[neuronindex][lastsecondfilter]/br.nS)
        plt.xlabel('time (s)')
        plt.ylabel('gI (nS)')
        plt.title(f'{N_inputs_list[neuronindex]} x {input_sizes[neuronindex]/br.nS}nS')
        plt.ylim([0, 140])
        
        ax = subplot()
        # plot the V trace of the DCN neuron
        ax.plot(net['st_DCN'].t[lastsecondfilter]/br.second, net['st_DCN'].v[neuronindex][lastsecondfilter]/br.mV)
        for t in net['sp_DCN'].spike_trains()[neuronindex]:
            if net.t - t < br.second: plt.axvline(t/br.second, 0, 100, c='black')
        plt.title(f"mean DCN rate = {len(net['sp_DCN'].spike_trains()[neuronindex])/net.t}")
        plt.xlabel('time (s)')
        plt.ylabel('V (mV)')
        plt.ylim([-75, 0])
        
        ax = subplot()
        # plot the PC spike triggered average DCN firing rate 
        #hist, bins = STA_rate(ax, net, neuronindex)
        hist, bins = STA_rates[neuronindex]
        ax.stairs(hist/(hist[0] or 1), bins/br.ms)
        #plt.title(f"N_PC = {len(net['PC'])} \n T = {net.t}")
        plt.ylabel('spikes (norm.)')
        plt.xlabel('t (ms)')
        
        # plot the ISI distribution of the DCN
        ax = subplot()
        ISIs = np.diff(net['sp_DCN'].spike_trains()[neuronindex])
        ax.hist(ISIs/br.ms, 50)
        plt.title(f'CV = {np.std(ISIs)/np.mean(ISIs):.2f}')
        plt.xlabel('DCN ISI (ms)')
        if 'ISI_xlim_ms' in kwargs: plt.xlim(kwargs['ISI_xlim_ms'])
        
        # # ISI dist of a PC
        # ax = subplot()
        # ISIs = np.diff(net['sp_PC'].spike_trains()[neuronindex])
        # ax.hist(ISIs/br.ms, 50)
        # plt.title(f'CV = {np.std(ISIs)/np.mean(ISIs):.2f}')
        # plt.xlabel('PC ISI (ms)')
        # if 'ISI_xlim_ms' in kwargs: plt.xlim(kwargs['ISI_xlim_ms'])
        
        ax = subplot()
    
    firing_rates = [len(tr)/net.t for tr in net['sp_DCN'].spike_trains().values()]
    
    gI = net['st_DCN'].gI
    gI_CV = np.std(gI, axis=1) / np.mean(gI, axis=1)
    
    # ax = subplot()
    ax.plot(input_sizes/br.nS, gI_CV)
    plt.xlim([0, 40])
    plt.ylim([0, 0.5])
    plt.xlabel('Input size (nS)')
    plt.ylabel('gI CV')
    ax = subplot()
    ax.plot(input_sizes/br.nS, firing_rates/br.Hz)
    plt.xlim([0, 40])
    plt.ylim([0, 120])
    plt.xlabel('Input size (nS)')
    plt.ylabel('Firing rate (Hz)')
    ax = subplot()
    ax.plot(gI_CV, firing_rates/br.Hz)
    plt.xlim([0, .5])
    plt.ylim([0, 120])
    plt.xlabel('gI CV')
    plt.ylabel('Firing rate (Hz)')
    
    plt.tight_layout()
    #plt.savefig(f'plots/{N_PC}x{abs(J_PC/br.mV):.2f}mV_{net.t/br.second:.0f}s.pdf')
    
    if kwargs.get('net', False): return net
    else: plt.show()
    
def spike_crosscorrelogram(in_ts, out_ts, window=10*br.ms):
    in_ts_ms = np.array(in_ts/br.ms)
    out_ts_ms = np.array(out_ts/br.ms)
    window_ms = window/br.ms
    shiftedtimes = np.concatenate([out_t-in_ts_ms[abs(in_ts_ms-out_t)<window_ms]
                                   for out_t in out_ts_ms] or [[]]) * br.ms
    return np.histogram(shiftedtimes, np.linspace(-window, window, 100))

def fig5(**kwargs):
    '''
    16 x 3nS, 10 x 10nS, 2 x 30nS
    vary the firing rate of the small, medium, large inputs by a multiplicative factor
    PC cell indices:
        (16 + 10 + 2) x N_factors
    DCN cell indices:
        (1 (small varied) + 1 (medium varied) + 1 (large varied)) x N_factors
        N_factors (rates of small inputs varied) + N_factors (medium) + N_factors (large)
    '''
    
    factor_list = kwargs.get('factor_list', np.linspace(0, 2, kwargs.get('N_factors', 5)))
    N_factors = len(factor_list)  # should be an odd number
    unity_factorindex = list(factor_list).index(1)
    
    # Use as PC_indexmap[factor, size_index], returns list
    # this gives the indices of the PCs which project onto
    PC_indexmap = [[range(28*j_factor + N_offset, 28*j_factor + N_offset + N_size)
                    for N_size, N_offset in [(16, 0), (10, 16), (2, 26)]]
                   for j_factor in range(N_factors)]
    
    connect = {'i': [], 'j': []}
    w = []
    for j_factor in range(N_factors):
        for j_size in range(3):
            j = j_size*N_factors + j_factor
            for i_size, size in enumerate([3, 10, 30]*br.nS):
                i_iter = (PC_indexmap[j_factor][j_size] 
                          if i_size == j_size
                          else PC_indexmap[unity_factorindex][i_size])
                connect['i'].extend(i_iter)
                connect['j'].extend([j for _ in range(len(i_iter))])
                w.extend([size for _ in range(len(i_iter))])
    
    kwargs['N_PC'] = (16+10+2)*N_factors
    kwargs['N_DCN'] = 3*N_factors
    kwargs['w'] = w
    if 'N_extE_DCN' not in kwargs: kwargs['N_extE_DCN'] = 4000
    
    kwargs['connect'] = connect
    
    if 'T' not in kwargs: kwargs['T'] = 1*br.second
    PC_indices = []
    PC_times = []
    for i, factor in enumerate(factor_list):
        if factor == 0: continue
        PC_indices_part, PC_times_part = fitted_PC_spikes(28, kwargs['T'], 28*i, 1/(factor*80*br.Hz))
        PC_indices.extend(PC_indices_part)
        PC_times.extend(PC_times_part/br.ms)
    PC_indices = np.take_along_axis(np.array(PC_indices), np.argsort(PC_times), axis=0)
    PC_times = np.sort(PC_times) * br.ms
    kwargs['PC_indices'] = PC_indices
    kwargs['PC_times'] = PC_times
    
    net = run_DCN_conductance(**kwargs)
    
    # get the inhibitory conductance CV
    gI = net['st_DCN'].gI
    gI_CV = np.std(gI, axis=1) / np.mean(gI, axis=1)
    
    plt.figure(figsize=(10,10))
    global plot_dims
    plot_dims = (2,3)
    ax = subplot(newplot=True)
    
    # ax = subplot(newplot=True)
    # plot firing rate vs input rate of varied cells
    firing_rate = [len(tr)/net.t for tr in net['sp_DCN'].spike_trains().values()]
    input_rate = [factor*80*br.Hz for j_size in range(3) for factor in factor_list]
    plt.plot(np.reshape(input_rate, [3,N_factors]).T, np.reshape(firing_rate, [3,N_factors]).T, 'o-')
    plt.legend(['small', 'medium', 'large'])
    plt.xlim([0, 160])
    plt.ylim([0, 200])
    plt.xlabel('input rate (Hz)')
    plt.ylabel('DCN rate (Hz)')
    
    ax = subplot()
    
    # plot firing rate against gI mean
    plt.plot(np.reshape(np.mean(gI, axis=1)/br.nS, [3,N_factors]).T, np.reshape(firing_rate, [3,N_factors]).T, 'o-')
    plt.legend(['small', 'medium', 'large'])
    plt.xlim([20, 80])
    plt.ylim([0, 200])
    plt.xlabel('gI mean (nS)')
    plt.ylabel('DCN rate (Hz)')
    
    ax = subplot()
    
    # plot firing rate against gI CV
    plt.plot(np.reshape(gI_CV, [3,N_factors]).T, np.reshape(firing_rate, [3,N_factors]).T, 'o-')
    plt.legend(['small', 'medium', 'large'])
    plt.xlim([0, 0.5])
    plt.ylim([0, 200])
    plt.xlabel('gI CV')
    plt.ylabel('DCN rate (Hz)')
    
    ax = subplot()
    
    # plot firing rate against gI std
    plt.plot(np.reshape(np.std(gI, axis=1)/br.nS, [3,N_factors]).T, np.reshape(firing_rate, [3,N_factors]).T, 'o-')
    plt.legend(['small', 'medium', 'large'])
    # plt.xlim([20, 80])
    # plt.ylim([0, 200])
    plt.xlabel('gI std (nS)')
    plt.ylabel('DCN rate (Hz)')
    
    plt.tight_layout()
    
    if kwargs.get('net', False): return net
    else: plt.show()


def fitted_PC_spikes(N_PC, T, startindex=0, mean_PC_ISI=1/(80*br.Hz)):
    sd_PC_ISI = -0.0015429*br.second + 0.58358*mean_PC_ISI  # fit from shuting
    if sd_PC_ISI <= 0*br.second: return ([], [])
    sigma_lognorm = np.sqrt(np.log(1 + (sd_PC_ISI**2 / mean_PC_ISI**2))) # dimensionless
    # mu_lognorm is shuting's and wikipedia's convention
    exp_mu_lognorm = mean_PC_ISI**2 / np.sqrt(mean_PC_ISI**2 + sd_PC_ISI**2) # has units of mean_PC_ISI
    dist_PC_ISI_unit = br.ms
    dist_PC_ISI = scipy.stats.lognorm(s=sigma_lognorm, scale=exp_mu_lognorm/dist_PC_ISI_unit)
    return generate_PC_spikes(dist_PC_ISI, dist_PC_ISI_unit, N_PC, T, startindex)

def generate_PC_spikes(ISI_dist, ISI_dist_unit, N_PC, T, startindex=0, chunk=100):
    ISIs = ISI_dist.rvs(size=[N_PC, chunk])
    while min(np.sum(ISIs, axis=1)) * ISI_dist_unit < T:
        ISIs = np.concatenate((ISIs, ISI_dist.rvs(size=[N_PC, chunk])), axis=1)
    PC_times = np.cumsum(ISIs, axis=1).flatten() * ISI_dist_unit
    PC_indices = np.full_like(ISIs.T, range(startindex, startindex+N_PC), dtype=int).T.flatten()
    # sort
    PC_indices = np.take_along_axis(PC_indices, np.argsort(PC_times), axis=0)
    PC_times = np.sort(PC_times)
    return PC_indices, PC_times

def run_DCN_conductance(**kwargs):    
    T = kwargs.get('T', 1*br.second)
    
    if 'PC_indices' not in kwargs:
        N_PC = kwargs.get('N_PC', 1)
        PC_indices, PC_times = fitted_PC_spikes(N_PC, T, mean_PC_ISI=kwargs.get('mean_PC_ISI', 1/(80*br.Hz)))
    else:
        PC_indices = kwargs.get('PC_indices', [0])
        PC_times = kwargs.get('PC_times', [0]*br.second)
        N_PC = max(PC_indices) + 1
    PC = br.SpikeGeneratorGroup(N_PC, PC_indices, PC_times)
    
    tau = kwargs.get('tau', 2.9*br.ms)
    tau_rp = kwargs.get('tau_rp', '(2+0*rand())*ms')#2*br.ms)
    theta = kwargs.get('theta', -50*br.mV)
    Vr = kwargs.get('Vr', -65*br.mV)  # check -50 to -60
    gL = 8.8*br.nS
    C_m = 70*br.pF
    V_L = -40*br.mV  # check: -40 or 0
    V_E = 0*br.mV
    V_I = -75*br.mV
    
    eqs = '''
        dv/dt = (gL*(V_L-v) + gE*(V_E-v) + gI*(V_I-v)) / C_m : volt (unless refractory)
        dgE/dt = -gE / tauE : siemens
        dgI/dt = -gI / tauI : siemens
        tauE : second
        tauI: second
    '''
    
    N_DCN = kwargs.get('N_DCN', 1)
    DCN = br.NeuronGroup(N_DCN,
                         eqs,
                         threshold='v>theta',
                         reset='v=Vr',
                         refractory=tau_rp,
                         method='euler',
                         name='DCN')
    DCN.v = Vr
    if 'gI' in kwargs: DCN.gI = kwargs['gI']
    DCN.tauE = kwargs.get('tauE', tau)
    DCN.tauI = kwargs.get('tauI', tau)
    extE_DCN = br.PoissonInput(DCN, 'gE', N=kwargs.get('N_extE_DCN', 0), rate=1*br.Hz, weight=1*br.nS)
    
    S = br.Synapses(PC, DCN, 'w : siemens', on_pre='gI += w', name='S')
    S.connect(**kwargs.get('connect', {}))#p=40/N_PC)
    S.w = kwargs.get('w', 1*br.nS)

    sp_DCN = br.SpikeMonitor(DCN, name='sp_DCN')
    st_DCN = br.StateMonitor(DCN, ['v','gE','gI'], record=True, name='st_DCN')
    sp_PC = br.SpikeMonitor(PC, name='sp_PC')

    net = br.Network(PC, DCN, extE_DCN, S, sp_DCN, sp_PC, st_DCN)    
    net.run(T, report='stderr')
    return net


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        print('standalone engine')
        br.set_device('cpp_standalone', build_on_run=True)
        br.prefs.devices.cpp_standalone.openmp_threads = 16
        result = globals()[sys.argv[1]](*sys.argv[2:])
    else:
        print('numpy engine')
        br.prefs.codegen.target = 'numpy'  # use the Python fallback

