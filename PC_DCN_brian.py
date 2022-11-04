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

from functools import partial

def starstarmap(p, f, kwargs_list):
    return p.map(partial(apply_kwargs, f), kwargs_list)

def apply_kwargs(f, kwargs):
    return f(**kwargs)



def fig2():
    pass    

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
    if kwargs.get('parallel', False):
        from multiprocessing import Pool
        with Pool(1) as p:
            STA_rates = dict(zip(neuronindices,
                                 p.map(partial(STA_rate,
                                               np.array(net['S'].i),
                                               np.array(net['S'].j),
                                               net['sp_PC'].spike_trains(),
                                               net['sp_DCN'].spike_trains()),
                                       neuronindices)))
    else:
        STA_rates = dict(zip(neuronindices,
                             map(partial(STA_rate,
                                           np.array(net['S'].i),
                                           np.array(net['S'].j),
                                           net['sp_PC'].spike_trains(),
                                           net['sp_DCN'].spike_trains()),
                                   neuronindices)))
    
    lastsecondfilter = net.t - net['st_DCN'].t < br.second
    
    plt.figure(figsize=(10,10))
    global plot_dims
    plot_dims = (7,4)
    ax = subplot(newplot=True)
    
    for neuronindex in neuronindices:
        # plot the total inhibitory conductance
        plt.plot(net['st_DCN'].t[lastsecondfilter]/br.second, net['st_DCN'].gI[neuronindex][lastsecondfilter]/br.nS)
        plt.xlabel('time (s)')
        plt.ylabel('I conductance (nS)')
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
        ax.stairs(hist, bins/br.ms)
        #plt.title(f"N_PC = {len(net['PC'])} \n T = {net.t}")
        plt.ylabel('spike-triggered output rate')
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

def STA_rate(S_i, S_j, PC_sp_tr_dict, DCN_sp_tr_dict, DCN_neuronindex=0, window=9.9*br.ms):
    '''
    S_i = np.array(net['S'].i)
    S_j = np.array(net['S'].j)
    PC_sp_tr_dict = net['sp_PC'].spike_trains()
    DCN_sp_tr_dict = net['sp_DCN'].spike_trains()
    '''
    # for DCN_neuronindex in range(len(net['DCN'])):
    PC_neuronindices = np.array(S_i)[np.array(S_j)==DCN_neuronindex]
    bins = np.linspace(-window, window, 100)
    hist = np.zeros(len(bins)-1, dtype='int32')
    DCN_sp_tr = DCN_sp_tr_dict[DCN_neuronindex]
    for PC_neuronindex in tqdm(PC_neuronindices):
        for sp_t in PC_sp_tr_dict[PC_neuronindex]:
            hist += np.histogram(DCN_sp_tr, bins + sp_t)[0]
            #print(hist)
    return hist, bins



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
    
    N_factors = kwargs.get('N_factors', 5)  # should be an odd number
    
    # Use as PC_indexmap[factor, size_index], returns list
    PC_indexmap = [[range(28*factor + N_offset, 28*factor + N_offset + N_size)
                    for N_size, N_offset in [(16, 0), (10, 16), (2, 26)]]
                   for factor in range(N_factors)]
    # print(PC_indexmap)
    
    connect = {'i': [], 'j': []}
    w = []
    for j_factor in range(N_factors):
        for j_size in range(3):
            j = j_size*N_factors + j_factor
            for i_size, size in enumerate([3, 10, 30]*br.nS):
                i_iter = (PC_indexmap[j_factor][j_size] 
                          if i_size == j_size
                          else PC_indexmap[int((N_factors-1)/2)][i_size])
                connect['i'].extend(i_iter)
                connect['j'].extend([j for _ in range(len(i_iter))])
                w.extend([size for _ in range(len(i_iter))])
    
    kwargs['N_PC'] = (16+10+2)*N_factors
    kwargs['N_DCN'] = 3*N_factors
    kwargs['w'] = w
    
    kwargs['connect'] = connect
    
    if 'T' not in kwargs: kwargs['T'] = 1*br.second
    PC_indices = []
    PC_times = []
    for i, factor in enumerate(np.linspace(0, 2, N_factors)):
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
    
    # ax = subplot(newplot=True)
    # plot firing rate vs input rate
    
    # plot firing rate against gI CV
    firing_rate = [len(tr)/net.t for tr in net['sp_DCN'].spike_trains().values()]
    plt.plot(np.reshape(gI_CV, [3,N_factors]).T, np.reshape(firing_rate, [3,N_factors]).T, 'o-')
    plt.legend(['small', 'medium', 'large'])
    plt.xlim([0, 0.5])
    plt.ylim([0, 200])
    
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

