import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

import brian2 as br

def plot_fit(ax, data, dist):
    ax.hist(data, 50, density=True)
    x = np.linspace(0, max(data), 100)
    param = dist.fit(data)#scipy.stats.fit(dist, data)
    fit = dist.pdf(x, *param)
    ax.plot(x, fit, 'r-')

plot_dims = (3,2)
def subplot(newplot=False):
    global plot_counter
    if newplot: plot_counter = 0
    plot_counter += 1
    return plt.subplot(*plot_dims, plot_counter)

def execfn(*args):
    exec(' '.join(args))

from functools import partial

def starstarmap(p, f, kwargs_list):
    return p.map(partial(apply_kwargs, f), kwargs_list)

def apply_kwargs(f, kwargs):
    return f(**kwargs)
    

def get_fig4(T_second=1):
    T = float(T_second)*br.second
    from multiprocessing import Pool
    params = [[400, -0.05*br.mV],
              [80, -0.25*br.mV],
              [40, -0.5*br.mV],
              [20, -1*br.mV],
              [10, -2*br.mV],
              [5, -4*br.mV]]
    paramsets = [dict(zip(['N_PC', 'J_PC'], val),
                      T=T, ISI_xlim_ms=[0, 150], input_ylim_mV_s=[0, 4000]
                      ) for val in params]
    print(paramsets)
    with Pool(min(len(params), 16)) as p:
        starstarmap(p, fig4, paramsets)

def fig4(**kwargs):
    
    mean_PC_ISI = kwargs.get('mean_PC_ISI', 1/(80*br.Hz))
    sd_PC_ISI = -0.0015429*br.second + 0.58358*mean_PC_ISI  # fit from shuting
    
    # dimensionless
    sigma_lognorm = np.sqrt(np.log(1 + (sd_PC_ISI**2 / mean_PC_ISI**2)))
    
    # has units of mean_PC_ISI
    exp_mu_lognorm = mean_PC_ISI**2 / np.sqrt(mean_PC_ISI**2 + sd_PC_ISI**2) # mu_lognorm is shuting's and wikipedia's convention
    
    dist_PC_ISI_unit = br.ms
    dist_PC_ISI = scipy.stats.lognorm(s=sigma_lognorm, scale=exp_mu_lognorm/dist_PC_ISI_unit)
    
    #print(dist_PC_ISI.rvs(10))
    #plt.hist(dist_PC_ISI.rvs(1000), 100)
    
    
    
    N_PC = kwargs.get('N_PC', 5)
    kwargs['w'] = 40*br.nS
    T = kwargs.get('T', 1*br.second)
    
    PC_indices, PC_times = generate_PC_spikes(dist_PC_ISI, dist_PC_ISI_unit, N_PC, T)
    kwargs['PC_indices'] = PC_indices
    kwargs['PC_times'] = PC_times
    
    net = run_DCN_conductance(**kwargs)
    
    plt.figure(figsize=(10,10))
    ax = subplot(newplot=True)
    
    
    # # plot the binned firing rate of the inhibitory inputs to the DCN
    # values, bins = np.histogram(np.concatenate([(tr/br.second)[net.t-tr<br.second] for tr in net['sp_PC'].spike_trains().values()]),
    #                             50)
    # ax.stairs(values*abs(J_PC)/br.mV/(bins[1]-bins[0]), bins)
    # plt.title(f'{N_PC} x {J_PC/br.mV} mV')
    # plt.xlabel('time (s)')
    # plt.ylabel('inhibitory synaptic input (mV/s)')
    # if 'input_ylim_mV_s' in kwargs: plt.ylim(kwargs['input_ylim_mV_s'])
    
    lastsecondfilter = net.t - net['st_DCN'].t < br.second
    
    # plot the total inhibitory conductance
    plt.plot(net['st_DCN'].t[lastsecondfilter]/br.second, net['st_DCN'].gI[0][lastsecondfilter]/br.nS)
    plt.xlabel('time (s)')
    plt.ylabel('I conductance (nS)')
    
    ax = subplot()
    # plot the V trace of the DCN neuron
    ax.plot(net['st_DCN'].t[lastsecondfilter]/br.second, net['st_DCN'].v[0][lastsecondfilter]/br.mV)
    for t in net['sp_DCN'].spike_trains()[0]:
        if net.t - t < br.second: plt.axvline(t/br.second, 0, 100, c='black')
    plt.title(f"mean DCN rate = {len(net['sp_DCN'])/len(net['DCN'])/net.t}")
    plt.xlabel('time (s)')
    plt.ylabel('V (mV)')
    plt.ylim([-10, 50])
    
    ax = subplot()
    # plot the PC spike triggered average DCN firing rate 
    plot_STA_rate(ax, net)
    
    # plot the ISI distribution of the DCN
    ax = subplot()
    ISIs = np.diff(net['sp_DCN'].spike_trains()[0])
    ax.hist(ISIs/br.ms, 50)
    plt.title(f'CV = {np.std(ISIs)/np.mean(ISIs):.2f}')
    plt.xlabel('DCN ISI (ms)')
    if 'ISI_xlim_ms' in kwargs: plt.xlim(kwargs['ISI_xlim_ms'])
    
    # ISI dist of a PC
    ax = subplot()
    ISIs = np.diff(net['sp_PC'].spike_trains()[0])
    ax.hist(ISIs/br.ms, 50)
    plt.title(f'CV = {np.std(ISIs)/np.mean(ISIs):.2f}')
    plt.xlabel('PC ISI (ms)')
    if 'ISI_xlim_ms' in kwargs: plt.xlim(kwargs['ISI_xlim_ms'])
    
    plt.tight_layout()
    #plt.savefig(f'plots/{N_PC}x{abs(J_PC/br.mV):.2f}mV_{net.t/br.second:.0f}s.pdf')
    
    if kwargs.get('net', False): return net


def plot_STA_rate(ax, net, window=9.9*br.ms):
    for DCN_neuronindex in range(len(net['DCN'])):
        PC_neuronindices = np.array(net['S'].i)[np.array(net['S'].j)==DCN_neuronindex]
        bins = np.linspace(-window, window, 100)
        hist = np.zeros(len(bins)-1, dtype='int32')
        DCN_sp_tr = net['sp_DCN'].spike_trains()[DCN_neuronindex]
        for PC_neuronindex in tqdm(PC_neuronindices):
            for sp_t in net['sp_PC'].spike_trains()[PC_neuronindex]:
                hist += np.histogram(DCN_sp_tr, bins + sp_t)[0]
                #print(hist)
        ax.stairs(hist, bins/br.ms)
        #plt.title(f"N_PC = {len(net['PC'])} \n T = {net.t}")
        plt.ylabel('spike-triggered output rate')
        plt.xlabel('t (ms)')


def generate_PC_spikes(ISI_dist, ISI_dist_unit, N_PC, T, chunk=100):
    ISIs = ISI_dist.rvs(size=[N_PC, chunk])
    while min(np.sum(ISIs, axis=1)) * ISI_dist_unit < T:
        ISIs = np.concatenate((ISIs, ISI_dist.rvs(size=[N_PC, chunk])), axis=1)
    PC_spikes = np.cumsum(ISIs, axis=1).flatten()
    PC_indices = np.full_like(ISIs.T, range(N_PC)).T.flatten()
    PC_indices = np.take_along_axis(PC_indices, np.argsort(PC_spikes), axis=0)
    PC_times = np.sort(PC_spikes)*ISI_dist_unit
    return PC_indices, PC_times


def fig5(T=1*br.second):
    N_DCN = 3*5
    w = np.concatenate([
        [np.repeat([mult*3,10,30], [16,10,2]) for mult in np.linspace(0,2,5)],
        [np.repeat([3,mult*10,30], [16,10,2]) for mult in np.linspace(0,2,5)],
        [np.repeat([3,10,mult*30], [16,10,2]) for mult in np.linspace(0,2,5)]
        ]).flatten() * br.nS    
    
    net = run_DCN_conductance(w=w, N_DCN=N_DCN, N_PC=16+10+2, T=T)
    
    # get the inhibitory conductance CV
    gI = net['st_DCN'].gI
    gI_CV = np.std(gI, axis=1) / np.mean(gI, axis=1)
    
    # plot firing rate against gI CV
    firing_rate = [len(tr)/net.t for tr in net['sp_DCN'].spike_trains().values()]
    plt.plot(np.reshape(gI_CV, [3,5]), np.reshape(firing_rate, [3,5]), 'o')
    
    plt.show()
    
    return net

def run_DCN_conductance(**kwargs):
    
    #conductance_wave = kwargs.get('conductance_wave', [])
    
    T = kwargs.get('T', 1*br.second)
    
    if 'PC_indices' not in kwargs:
        N_PC = kwargs.get('N_PC', 1)
        
        mean_PC_ISI = kwargs.get('mean_PC_ISI', 1/(80*br.Hz))
        sd_PC_ISI = -0.0015429*br.second + 0.58358*mean_PC_ISI  # fit from shuting
        
        # dimensionless
        sigma_lognorm = np.sqrt(np.log(1 + (sd_PC_ISI**2 / mean_PC_ISI**2)))
        
        # has units of mean_PC_ISI
        exp_mu_lognorm = mean_PC_ISI**2 / np.sqrt(mean_PC_ISI**2 + sd_PC_ISI**2) # mu_lognorm is shuting's and wikipedia's convention
        
        dist_PC_ISI_unit = br.ms
        dist_PC_ISI = scipy.stats.lognorm(s=sigma_lognorm, scale=exp_mu_lognorm/dist_PC_ISI_unit)
        
        PC_indices, PC_times = generate_PC_spikes(dist_PC_ISI, dist_PC_ISI_unit, N_PC, T)
    else:
        PC_indices = kwargs.get('PC_indices', [0])
        PC_times = kwargs.get('PC_times', [0]*br.second)
        N_PC = len(set(PC_indices))
        
    PC = br.SpikeGeneratorGroup(N_PC, PC_indices, PC_times)
    
    tau = kwargs.get('tau', 2.9*br.ms)
    tau_rp = kwargs.get('tau_rp', '(1+2*rand())*ms')#2*br.ms)
    
    theta = kwargs.get('theta', 20*br.mV)
    Vr = kwargs.get('Vr', 10*br.mV)
    
    N_DCN = kwargs.get('N_DCN', 1)
    
    gL = 100*br.nS
    
    C_m = 1*br.nF
    V_L = 0*br.mV
    V_E = 40*br.mV
    V_I = -20*br.mV
    
    eqs = '''
        dv/dt = (gL*(V_L-v) + gE*(V_E-v) + gI*(V_I-v)) / C_m : volt (unless refractory)
        dgE/dt = -gE / tau : siemens
        dgI/dt = -gI / tau : siemens
    '''
    
    DCN = br.NeuronGroup(N_DCN,
                         eqs,
                         threshold='v>theta',
                         reset='v=Vr',
                         refractory=tau_rp,
                         method='euler',
                         name='DCN')
    
    
    # extE_PC = br.PoissonInput(PC, 'v', N=200, rate=10*br.Hz, weight=1*br.mV)
    # extI_PC = br.PoissonInput(PC, 'v', N=200, rate=10*br.Hz, weight=-1*br.mV)
    extE_DCN = br.PoissonInput(DCN, 'gE', N=kwargs.get('N_extE_DCN', 60000), rate=1*br.Hz, weight=1*br.nS)
    
    S = br.Synapses(PC, DCN, 'w : siemens', on_pre='gI += w', name='S')
    S.connect()#p=40/N_PC)
    S.w = kwargs.get('w', 1*br.nS)

    sp_DCN = br.SpikeMonitor(DCN, name='sp_DCN')
    st_DCN = br.StateMonitor(DCN, ['v','gE','gI'], record=True, name='st_DCN')

    sp_PC = br.SpikeMonitor(PC, name='sp_PC')
    # st_PC = br.StateMonitor(PC, 'v', record=True, name='st_PC')
    
    # net = br.Network(PC, DCN,
    #                  extE_PC, extI_PC, extE_DCN,
    #                  S,
    #                  sp_DCN, st_DCN, sp_PC, st_PC)

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
    
    
    
    
    
    
    
    