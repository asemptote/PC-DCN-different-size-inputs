import numpy as np
import scipy
import brian2.only as br
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import network
import analysis

from functools import partial
from multiprocessing import Manager, Pool
import sys
from time import time
import os

PLOTNAME = 'figs/20230109'

FINAL_PARAMS = {'C_m': 50*br.pF,
                'N_extE_DCN': 20000,
                # 'tau_rp': 1*br.ms,
                }

G_nS = np.genfromtxt('data/G.csv') * 0.4 / 2.3  # nS


import pickle



# for /l %i in (1 1 10) do python fig2.py save_xcorr_parallel(seed=%i, nreps=160)

def save_xcorr_parallel(**kwargs):
    kwargs = {**{
        'T': 10*br.second,  # max: 100
        'N_DCN': 100,  # max: 100
        }, **kwargs}
    totalconductance_nS = kwargs.get('totalconductance', 200*br.nS)/br.nS
    
    seed = kwargs.get('seed', 0)
    if 'input_sizes' not in kwargs:
        with analysis.seed(seed):
            input_sizes_nS = []
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
            input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
            input_sizes_nS.sort()
            kwargs['input_sizes'] = input_sizes_nS*br.nS
    
    nreps = kwargs.get('nreps', 16)
    with Pool(16, initializer=np.random.seed) as p:
        hist_lists, bins_list = zip(*tqdm(analysis.starstar(p.imap,
                                                            run_xcorr,
                                                            [kwargs for _ in range(nreps)])))
    fname = kwargs.get('fname', f'{PLOTNAME}_fig2_xcorr_{seed}.dat')
    try:
        with open(fname, 'rb') as f:
            hist_list_f, bins_f, input_sizes_f, totaltime_s = pickle.load(f)
        assert sorted(input_sizes_f) == sorted(kwargs['input_sizes'])
    except FileNotFoundError:
        hist_list_f = None
        totaltime_s = 0
    hist_list = np.sum(hist_lists, axis=0)
    if hist_list_f is not None: hist_list = hist_list + hist_list_f
    bins = bins_list[0]
    totaltime_s += nreps * (kwargs['T']/br.second) * kwargs['N_DCN']
    print(f'total time: {totaltime_s} s')
    with open(fname, 'wb') as f:
        pickle.dump((hist_list, bins, kwargs['input_sizes'], totaltime_s), f)
    
    
    
    

def run_xcorr(**kwargs):
    '''
    returns the crosscorrelogram
    '''
    kwargs = {**{
        'record': False,
        'mean_PC_ISI': 1/(83*br.Hz),
        'report': None,
        'out': {
            'PC_trs': "net['sp_PC'].spike_trains()",
            'DCN_trs': "net['sp_DCN'].spike_trains()",
            'S_i': "connect['i']",
            'S_j': "connect['j']",
            # 'T': 'net.t',
            # 'sizes': "kwargs['input_sizes']",
            },
        }, **FINAL_PARAMS, **kwargs}
    out = run_differentsizeinputs(**kwargs)
    print(np.mean([len(tr)/tr[-1] for tr in out['DCN_trs'].values()]))
    hist_list, bins = average_xcorr(out['PC_trs'],
                                    out['DCN_trs'],
                                    out['S_i'],
                                    out['S_j'])
    return hist_list, bins
    
    



def populate_df(df_data, hist, bin_centres, avg_window):
    '''
    side effect: modifies `df_data`
    '''
    baseline = np.mean([hist[i] for i in range(avg_window)]) or 1
    c_plus = np.mean(hist[[len(hist)//2-i for i in range(avg_window)]])
    df_data['$c_+$ (norm.)'].append(c_plus/baseline)
    c_minus = np.mean(hist[[len(hist)//2+1+i for i in range(avg_window)]])
    df_data['$c_-$ (norm.)'].append(c_minus/baseline)
    # gap = lambda ts: max(ts, default=0)-min(ts, default=0)
    # df_data[r'$w_{0.5}$ (ms)'].append(gap(bin_centres[(hist<.5*baseline) & (bin_centres>0)]*1000))
    # df_data[r'$w_0$ (ms)'].append(gap(bin_centres[(hist==0) & (bin_centres>0)]*1000))
    w_supp_idx = 0
    bin_idx = np.argwhere((hist<(c_minus+baseline)/2) & (bin_centres>0))[0,0]
    while hist[bin_idx+w_supp_idx] < (c_minus+baseline)/2:
        w_supp_idx += 1
    w_supp = bin_centres[bin_idx+w_supp_idx] - bin_centres[bin_idx]
    df_data[r'$w_{supp}$ (ms)'].append(w_supp*1000)
    # df_data[r'$w_{supp}$ (ms)'].append(gap(bin_centres[(hist<(c_minus+baseline)/2) & (bin_centres>0)]*1000))
    return baseline
    

def plot_xcorr(**kwargs):
    
    import seaborn as sns
    
    # xcorr stats
    avg_window = kwargs.get('moving_average_window', 10)
    
    seeds = kwargs.get('seeds', range(1, 11))
    
    df_xcorr_data = {key: [] for key in [
        'seed',
        'input size (nS)',
        'time (ms)',
        'spike count',
        ]}
    
    df_data = {
        '$c_+$ (norm.)': [],
        '$c_-$ (norm.)': [],
        r'$w_{supp}$ (ms)': [],
        'input size (nS)': [],
        'seed': [],
        }
                     
    
    for seed in seeds:
        
        fname = f'{PLOTNAME}_fig2_xcorr_{seed}.dat'
        with open(fname, 'rb') as f:
            hist_list, bins, input_sizes, count = pickle.load(f)
        print(count)
        
        bin_centres = (bins[:-1] + bins[1:])/2
        
        for size_idx, size_nS in enumerate(input_sizes/br.nS):
            for spike_count, t_ms in zip(hist_list[size_idx], bin_centres*1000):
                df_xcorr_data['input size (nS)'].append(size_nS)
                df_xcorr_data['seed'].append(seed)
                df_xcorr_data['time (ms)'].append(t_ms)
                df_xcorr_data['spike count'].append(spike_count)
        
        # plot_traces(input_sizes=input_sizes, show_plot=False)
        
        if seed == 2:
            x = np.linspace(0, 200, 100)
            # plot the input cdfs
            input_sizes_nS = input_sizes/br.nS
            
            plt.figure(figsize=(6/2, 6/2))
            sns.ecdfplot(G_nS)
            sns.ecdfplot(input_sizes_nS)
            plt.xlabel('input size (nS)')
            plt.title(f'{len(input_sizes_nS)} inputs')
            plt.tight_layout()
            plt.savefig(f'{PLOTNAME}_fig2d_cdf.png')
        
            # plot xcorr fn
            plt.figure(figsize=(6/2, 6/2))
            
            bin_centres = (bins[:-1] + bins[1:])/2
            norm = mpl.colors.Normalize(vmin=0, vmax=max(input_sizes_nS))
            cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.turbo)
            cmap.set_array([])
            cb = plt.colorbar(cmap)
            for hist, size_nS in zip(hist_list, input_sizes_nS):
                # plt.stairs(hist/(hist[0] or 1), bins*1000, color=cmap.to_rgba(size_nS))
                plt.plot(bin_centres*1000, hist/(hist[0] or 1), color=cmap.to_rgba(size_nS))
                cb.ax.axhline(y=size_nS, c='w')
            plt.ylabel('spikes (norm.)')
            plt.xlabel('t (ms)')
            plt.ylim([0,2])
            plt.tight_layout()
            plt.savefig(f'{PLOTNAME}_fig2d.png')
        
        
        
        
        
        # other_seeds = range(1, 11)
        # for other_seed in other_seeds:
        #     if other_seed == seed: continue
        #     fname = f'{PLOTNAME}_fig2_xcorr_{other_seed}.dat'
        #     with open(fname, 'rb') as f:
        #         hist_list_other, bins_other, input_sizes_other, count_other = pickle.load(f)
        #     bin_centres_other = (bins_other[:-1] + bins_other[1:])/2
        #     print(f'seed {other_seed}: {count} s')
        #     for hist, size_nS in zip(hist_list_other, input_sizes_other/br.nS):
        #         populate_df(df_data, hist, size_nS, bin_centres_other)
        #         df_data['hue'].append(1)
        
        # bin_centres = (bins[:-1] + bins[1:])/2
        # for hist, size_nS in zip(hist_list, input_sizes_nS):
        #     populate_df(df_data, hist, size_nS, bin_centres)
        #     df_data['hue'].append(2)
        
        bin_centres = (bins[:-1] + bins[1:])/2
        for hist, size_nS in zip(hist_list, input_sizes/br.nS):
            populate_df(df_data, hist, bin_centres, avg_window)
            df_data['input size (nS)'].append(size_nS)
            df_data['seed'].append(seed)
    
    df_xcorr = pd.DataFrame(df_xcorr_data)
    df_xcorr.to_csv(f'{PLOTNAME}_fig2_xcorr.csv')    
    
    df = pd.DataFrame(df_data)
    df.to_csv(f'{PLOTNAME}_fig2_stats.csv')
    
    # plot the aggregate xcorr stats
    import seaborn as sns
    # g = sns.pairplot(df)#, hue='vary size', diag_kind='hist')
    # g.set(xlim=(0,None), ylim=(0,None))
    
    g = sns.PairGrid(df,
                      y_vars=['$c_+$ (norm.)', #'peak corr (norm.)',
                            '$c_-$ (norm.)', #'min corr (norm.)',
                            # r'$w_{0.5}$ (ms)', #'width at 0.5 (ms)',
                            # r'$w_0$ (ms)', #'width at 0 (ms)',
                            r'$w_{supp}$ (ms)', #width of suppression (ms)
                            ],
                      x_vars=['input size (nS)'],
                       height=1.5, aspect=2, hue='seed')
    g.map(sns.scatterplot, size=0.25)
    plt.savefig(f'{PLOTNAME}_fig2efg.png')
    
    plt.show()



# def plot_traces(**kwargs):
#     kwargs = {**{
#         'T': 1*br.second,
#         'N_DCN': 1,
#         'record': True,
#         'N_extE_DCN': 20000,
#         'mean_PC_ISI': 1/(83*br.Hz),
#         # 'moving_average_window': 10,
#         'report': 'stdout',
#         'complete_gI': True,
#         'C_m': fig2_C_m,
#         'out': {
#             'PC_trs': "net['sp_PC'].spike_trains()",
#             'DCN_trs': "net['sp_DCN'].spike_trains()",
#             'S_i': "connect['i']",
#             'S_j': "connect['j']",
#             'T': 'net.t',
#             'sizes': "kwargs['input_sizes']",
#             'V': "net['st_DCN'].v",
#             'gI_nS': 'gI.values.T * 1e9',
#             'gI_complete_nS': 'gI_array_complete_nS',
#             }
#         }, **kwargs}
    
#     sim = run_differentsizeinputs(**kwargs)
    
#     input_sizes_nS = sim['sizes']/br.nS
#     norm = mpl.colors.Normalize(vmin=0, vmax=max(input_sizes_nS))
#     cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.turbo)
#     cmap.set_array([])
    
#     # raster plot
#     plt.figure(figsize=(23/2, 4/2))
#     PC_trs = np.take_along_axis(np.array([sim['PC_trs'][i]/br.ms for i in range(len(input_sizes_nS))]), np.argsort(input_sizes_nS), axis=0)
#     for idx, size_nS in enumerate(input_sizes_nS):
#         plt.vlines(PC_trs[idx], idx, idx+1, cmap.to_rgba(size_nS))
#     plt.xlim([900, 1000])
#     plt.xlabel('time (ms)')
#     plt.ylabel('Neuron')
#     plt.tight_layout()
#     plt.savefig(f'{PLOTNAME}_fig2a.png')
    
#     # gI trace
#     plt.figure(figsize=(23/2, 4/2))
#     ts_gI_ms = np.linspace(0, sim['T']/br.ms, len(sim['gI_nS'][0]))
#     plt.plot(ts_gI_ms, sim['gI_nS'][0], 'black')
#     for idx, size_nS in enumerate(input_sizes_nS):
#         plt.plot(ts_gI_ms, sim['gI_complete_nS'][0][idx], color=cmap.to_rgba(size_nS))
#     plt.xlim([900, 1000])
#     plt.xlabel('time (ms)')
#     plt.ylabel('$g_I$ (nS)')
#     plt.tight_layout()
#     plt.savefig(f'{PLOTNAME}_fig2b.png')
    
#     # V trace
#     V_mV = sim['V'][0]/br.mV
#     plt.figure(figsize=(23/2, 4/2))
#     plt.plot(np.linspace(0, sim['T']/br.ms, len(V_mV)), V_mV)
#     # plt.vlines(sim['DCN_trs'][0]/br.ms, -75, 0, 'black')
#     for t_ms in sim['DCN_trs'][0]/br.ms:
#         plt.axvspan(t_ms, t_ms+2, color='grey', zorder=10)
#     plt.ylim([-75, 0])
#     plt.xlim([900, 1000])
#     plt.xlabel('time (ms)')
#     plt.ylabel('$V_m$ (mV)')
#     plt.title(f"rate = {len(sim['DCN_trs'][0])/sim['T']/br.Hz:.0f} Hz")
#     plt.tight_layout()
#     plt.savefig(f'{PLOTNAME}_fig2c.png')
    
#     if kwargs.get('show_plot', True): plt.show()
#     return sim


def average_xcorr(PC_spike_trains, DCN_spike_trains, S_i, S_j, p=None, m=None, chunksize=None):#parallel=False, chunksize=None):
    if p:#arallel:
        # m = Manager()
        in_tr_list = m.dict(PC_spike_trains)
        out_tr_list = m.dict(DCN_spike_trains)
    else:
        in_tr_list = PC_spike_trains
        out_tr_list = DCN_spike_trains
    # with (Pool(16) if parallel else analysis.yieldexpr(None)) as p:
    mapfn = ((partial(p.imap, chunksize=chunksize) if chunksize else p.imap)
             # if parallel else map)
             if p else map)
    hist_list, bins_list = zip(*tqdm(
        analysis.star(mapfn, 
                      partial(analysis.xcorr_tr, in_tr_list, out_tr_list), 
                      zip(S_i, S_j)), 
        total=len(S_i),
        disable=not bool(p),
    ))
    bins = bins_list[0]
    
    hist_list_avg = []
    N_inputs = len(PC_spike_trains)//len(DCN_spike_trains)
    for i_input in range(N_inputs):
        hist = np.sum([hist for i_S, hist in enumerate(hist_list) if S_i[i_S]%N_inputs==i_input], axis=0)
        hist_list_avg.append(hist)
    return hist_list_avg, bins

def run_differentsizeinputs(**kwargs):
    kwargs = {**{
        'input_sizes': np.concatenate([np.repeat(3, 16),
                        np.repeat(10, 10),
                        np.repeat(30, 2)])*br.nS,
        'N_DCN': 1,
        'N_extE_DCN': 5000,
        }, **kwargs}
    input_sizes = kwargs['input_sizes']
    N_inputs = len(input_sizes)
    kwargs['N_PC'] = N_inputs * kwargs['N_DCN']
    connect = {'i': [], 'j': []}
    kwargs['w'] = []
    for j_DCN in range(kwargs['N_DCN']):
        connect['i'].extend(range(N_inputs*j_DCN, N_inputs*(j_DCN+1)))
        connect['j'].extend([j_DCN for _ in input_sizes])
        kwargs['w'].extend(input_sizes)
    kwargs['connect'] = connect
    return network.run_DCN_wave(**kwargs)
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        result = eval(' '.join(sys.argv[1:]))