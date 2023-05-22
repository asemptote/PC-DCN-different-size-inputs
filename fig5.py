import numpy as np
import scipy
import brian2.only as br
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import network
import analysis
import fig2

from functools import partial
from multiprocessing import Manager, Pool
import sys
from time import time
import os
import pickle

PLOTNAME = 'figs/20230109'

G_nS = fig2.G_nS
FINAL_PARAMS = fig2.FINAL_PARAMS

def vary_individual_inputs(**kwargs):
    # the nth DCN neuron having the nth PC input vary its frequency
    # rotate the input sizes along the PCs for each DCN cell
    kwargs = {**{
        'out': {
            'DCN_rate_Hz': "[len(tr) for tr in net['sp_DCN'].spike_trains().values()]/net.t/br.Hz",
            'gI_mean_nS': "np.mean(gI.values.T*1e9, axis=1)",
            'gI_std_nS': "np.std(gI.values.T*1e9, axis=1)",
            },
        'report': None,
        'record': False,
        'T': 10*br.second,
        'nreps': 10,
        'nfactors': 16,
        'runsim': False,
        }, **kwargs}
        
    df_data = {key: [] for key in [
        'input rate (Hz)',
        '$g_I$ mean (nS)',
        '$g_I$ std (nS)',
        '$g_I$ CV',
        'input size (nS)',
        'output rate (Hz)',
        'seed',
        ]}
        
    if kwargs['runsim']:
        for seed in range(1,11):
            # generate the input sizes
            totalconductance_nS = kwargs.get('totalconductance', 200*br.nS)/br.nS
            with analysis.seed(seed):
                input_sizes_nS = []
                while sum(input_sizes_nS) < totalconductance_nS:
                    input_sizes_nS.append(np.random.choice(G_nS))
                input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
                input_sizes_nS.sort()
                kwargs['input_sizes'] = input_sizes_nS*br.nS
            
            N_inputs = len(input_sizes_nS)
            kwargs_list = []
            param_list = []
            factors = np.linspace(0, 2, kwargs['nfactors'])
            nreps = kwargs['nreps']
            for rep in range(nreps):
                for factor in factors:
                    kwargs_list.append({
                        **kwargs,
                        'input_rates': ([factor*80]+[80 for _ in range(N_inputs-1)])*br.Hz
                        })
            
            # print(kwargs_list)
            with Pool(16) as p:
                sim_list = list(tqdm(analysis.starstar(p.imap, run_ratevarying, kwargs_list),
                                     total=len(kwargs_list),
                                     disable=False,
                                     desc=f'seed {seed}'))
    
            for idx_factor, factor in enumerate(factors):
                sims = [sim_list[rep] for rep in range(idx_factor, len(sim_list), len(factors))]
                for idx_DCN in range(N_inputs):
                    df_data['input rate (Hz)'].append(factor*80)
                    df_data['$g_I$ mean (nS)'].append(
                        np.mean([sim['gI_mean_nS'][idx_DCN] for sim in sims])
                        )
                    df_data['$g_I$ std (nS)'].append(
                        np.mean([sim['gI_std_nS'][idx_DCN] for sim in sims])
                        )
                    df_data['$g_I$ CV'].append(
                        np.mean([sim['gI_std_nS'][idx_DCN]/sim['gI_mean_nS'][idx_DCN] for sim in sims])
                        )
                    df_data['input size (nS)'].append(input_sizes_nS[idx_DCN])
                    df_data['output rate (Hz)'].append(
                        np.mean([sim['DCN_rate_Hz'][idx_DCN] for sim in sims])
                        )
                    df_data['seed'].append(seed)

        df = pd.DataFrame(df_data)
        df.to_csv(f'{PLOTNAME}_fig5.csv')
        
        
    else:
        df = pd.read_csv(f'{PLOTNAME}_fig5.csv', index_col=0)
    # return df
    
    fig_seed = 8
    df_seed = df[df['seed'] == fig_seed]
    df_nans = pd.DataFrame({'input size (nS)': list(set(df_seed['input size (nS)'])),
                            'input rate (Hz)': np.nan})
    df_fig = (pd.concat([df_seed, df_nans])
              .sort_values(['input size (nS)', 'input rate (Hz)'])
              )
    df_fig.to_csv(f'{PLOTNAME}_fig5_seed_{fig_seed}.csv')
                  
    
    # compute the slope of the input-output curve
    # note: some of the inputs are the same size (double-drawn)
    #       rely on the ordering of the for loops above
    df_slope = (df
                .groupby(['input size (nS)', 'input rate (Hz)', 'seed'])
                .mean()
                .reset_index()
                .sort_values('input rate (Hz)')
                .groupby(['input size (nS)', 'seed'])
                .apply(lambda x: np.mean(np.diff(x['output rate (Hz)'])/np.diff(x['input rate (Hz)'])))
                .rename('slope')
                .reset_index()
                .sort_values(['seed', 'input size (nS)'])
                )
    df_slope.to_csv(f'{PLOTNAME}_fig5_slope.csv')
    plt.figure()
    sns.lineplot(df_slope,
                 x='input size (nS)',
                 y='slope',
                 marker='o',
                 style='seed',
                 dashes=False,
                 legend=False,
                 )
    
    g = sns.PairGrid(df[df['seed']==1],
                     x_vars=['input rate (Hz)',
                     '$g_I$ mean (nS)',
                     '$g_I$ std (nS)',
                     '$g_I$ CV',],
                     y_vars='output rate (Hz)',
                     hue='input size (nS)',
                     )
    g.map(sns.lineplot, size=0.25, sort=False)
    plt.show()
                     
    
    
def run_ratevarying(**kwargs):
    '''
    given a list of input sizes and a varying factor, check the effect of
    varying the rate of each of the input sizes while keeping the other input sizes
    at a fixed rate (80Hz)
    There are N_PC DCN cells, with the nth DCN cell having the nth input size's
    rate varied
    We implement this by rotating the input sizes along the PCs for each DCN cell,
    so that only the first PC's rate is varied
    '''
    kwargs = {**{
        'T': 1*br.second,
        # 'N_copies': 1,
        }, **FINAL_PARAMS, **kwargs}
    N_inputs = len(kwargs['input_sizes'])
    input_rates = kwargs.get('input_rates', [80*br.Hz for _ in range(N_inputs)])
    PC_indices = []
    PC_times = []
    for i_input, rate in enumerate(input_rates):
        try:
            PC_indices_part, PC_times_part = analysis.fitted_PC_spikes(
                # range(i_input, kwargs['N_copies']*N_inputs, N_inputs), 
                [i_input],
                kwargs['T'],
                1/rate
            )
        except ValueError:
            PC_indices_part = []
            PC_times_part = []
        PC_indices.extend(PC_indices_part)
        PC_times.extend(PC_times_part/br.ms)
    PC_indices = np.take_along_axis(np.array(PC_indices), np.argsort(PC_times), axis=0)
    PC_times = np.sort(PC_times) * br.ms
    kwargs['PC_indices'] = PC_indices
    kwargs['PC_times'] = PC_times
    
    input_sizes = kwargs['input_sizes']
    N_inputs = len(input_sizes)
    kwargs['N_PC'] = N_inputs# * kwargs['N_copies']
    kwargs['N_DCN'] = kwargs['N_PC']
    connect = {'i': [], 'j': []}
    kwargs['w'] = []
    
    for idx_PC in range(N_inputs):
        for idx_DCN in range(N_inputs):
            connect['i'].append(idx_PC)
            connect['j'].append(idx_DCN)
            kwargs['w'].append(input_sizes[(idx_PC+idx_DCN)%N_inputs])
    kwargs['connect'] = connect
    return network.run_DCN_wave(**kwargs)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        result = eval(' '.join(sys.argv[1:]))