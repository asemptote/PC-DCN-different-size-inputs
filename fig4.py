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

# PLOTNAME = 'figs/20221220_fits_'
PLOTNAME = 'figs/20230203'

EMPIRICAL_N_extE_DCN = {
    'E15_times_A': 2110423//90,
    'E20_times_A': 2813032//90,
    'E25_times_A': 3512220//90,
    'E30_times_A': 4220865//90,
    'E40_times_A': 5626325//90,
    'E50_times_A': 7030317//90,
    }

CELL_NAMES = ['0830_cell_1',
              '0830_cell_2',
              '0901_cell_1',
              '0901_cell_3',
              '0902_cell_1',
              '0902_cell_2',
              '0902_cell_4',
              ]

DF_EMPIRICAL = pd.read_csv('data/fig4_rates.csv')
DF_CELLRECS = {name: (pd.read_csv(f'data/{name}.csv')
                      .assign(trial_avg=lambda df_x: np.mean([df_x[col] for col in df_x if col.startswith('trial')], axis=0))
                      )
               for name in CELL_NAMES}
DF_CELLRECS_FILTERED = {name: (pd.read_csv(f'data/{name}.csv')
                               .filter(['size']+[col for col in DF_CELLRECS[name] if col.startswith('trial ') and (lambda x: list(x)==sorted(x) and len(x)==len(set(x)))(DF_CELLRECS[name].sort_values('size')[col])])
                               .assign(trial_avg=lambda df_x: np.mean([df_x[col] for col in df_x if col.startswith('trial')], axis=0))
                               )
                        for name in CELL_NAMES}
# DF_CELLRECS = DF_CELLRECS_FILTERED
DF_FIG4PARAMS = pd.read_csv('data/fig4_params.csv')
DF_AVERAGES = pd.read_csv('data/size++.txt') 

def fit_all_cells():
    # start /b to show it in the same window
    argstring = 'maxiter=100'
    avgtrials = True
    for cellname in CELL_NAMES:
        print(cellname)
        if avgtrials:
            os.system(f"start /min python fig4.py fit_uniforminputs(cellname='{cellname}', trial=None, {argstring})")
        else:
            trials = [int(key.split()[-1]) for key in DF_CELLRECS[cellname].keys()
                      if key != 'size']
            for trial in trials[:-1]:
                os.system(f"start /min python fig4.py fit_uniforminputs(cellname='{cellname}', trial={trial}, {argstring})")
            os.system(f"start /min /wait python fig4.py fit_uniforminputs(cellname='{cellname}', trial={trials[-1]}, {argstring})")

def plot_all_cells(**kwargs):
    import re
    from glob import glob
    avgtrials = True
    dfs = []
    dfs_gICV = []
    dfs_empirical_full = []
    kwargs['return_df'] = True
    kwargs['T'] = 10*br.second
    kwargs['input_range'] = list(range(4, 100)) + list(range(100, 1001, 100)) + [None]
    fnames = glob(f"{kwargs.get('plotname', PLOTNAME)}*.gif")
    # for cellname, trial_str, V_L_str, gL_str, Vr_str, niters_str, fname \
    #     in [re.findall('fits_(.*)_trial_(.*)_VLmV_(.*)_gLnS_(.*)_VrmV_(.*)_niters_(.*).gif', fname)[0]
    #         +(fname,) for fname in fnames]:
    for cellname, trial_str, V_L_str, gL_str, Vr_str, C_m_str, theta_str, niters_str, fname \
        in [re.findall('fits_(.*)_trial_(.*)_(.*)_(.*)_(.*)_(.*)_(.*)_niters_(.*).gif', fname)[0]
            +(fname,) for fname in fnames]:
        if (trial_str != 'None' and avgtrials) or (trial_str == 'None' and not avgtrials):
            continue
        print(cellname, trial_str)
        trial = None if avgtrials else int(trial_str)
        kwargs['V_L'] = float(V_L_str)*br.mV
        kwargs['gL'] = float(gL_str)*br.nS
        kwargs['Vr'] = float(Vr_str)*br.mV
        kwargs['C_m'] = float(C_m_str)*br.pF
        kwargs['theta'] = float(theta_str)*br.mV
        kwargs['cellname'] = cellname
        # niters = int(niters_str)
        df_params = pd.read_csv('data/fig4_params.csv')
        params = df_params[df_params['name']==kwargs['cellname']].iloc[0]
        # kwargs['C_m'] = params['C capacitance (pF)']*br.pF
        kwargs['N_extE_DCN'] = EMPIRICAL_N_extE_DCN[f"E{params['E conductance (nS)']}_times_A"]
        # kwargs['theta'] = params['firing threshold (mV)']*br.mV
        try:
            df = pd.read_csv(fname[:-4]+'.csv')
        except FileNotFoundError:
            df = plot_uniforminputs(**kwargs)
            df.to_csv(fname[:-4]+'.csv')
        df['cellname'] = cellname
        df['trial'] = trial
        dfs.append(df)
        dfs_gICV.append(df[['input size (nS)', r'$g_I$ CV']])
        df_empirical_full = DF_CELLRECS[cellname].melt('size',value_name='rate')
        df_empirical_full['cellname'] = cellname
        dfs_empirical_full.append(df_empirical_full)
    df = pd.concat(dfs)#.groupby(level=0)
    
    # for cellname in CELL_NAMES:
    #     df_avg = df[df['cellname']==cellname].drop(columns='trial').groupby(level=0).mean()
    #     # print(df_avg)
    #     # print(len(df_avg))
    #     # return
    #     df_cell = DF_CELLRECS[cellname]
    #     plt.plot(df_avg['input size (nS)'], df_avg['firing rate (Hz)'])
    #     for k in df_cell.keys():
    #         if k != 'size': 
    #             plt.plot(df_cell['size'], df_cell[k], color='grey')
    #     plt.scatter(DF_EMPIRICAL['size'], DF_EMPIRICAL[f"ave_{cellname}"])
    #     plt.title(cellname)
    #     plt.xlabel('input size (nS)')
    #     plt.ylabel('firing rate (Hz)')
    #     plt.savefig(f'20221214_{cellname}.png')
    #     plt.figure()
        
    # plt.show()
    # return
    df_empirical = pd.read_csv('data/size++.txt')    
    df_gICV = pd.concat(dfs_gICV).groupby(level=0).mean()
    # print(df_gICV)
    for row in df_gICV.iloc:
        df.loc[abs(df['input size (nS)']-row['input size (nS)'])<.001, r'$g_I$ CV'] = row[r'$g_I$ CV']
    df_empirical_full = pd.concat(dfs_empirical_full)
    df_empirical_avgs = df_empirical_full.groupby(['size', 'cellname']).mean()
    df_empirical_avgs = df_empirical_avgs.merge(df_empirical[['size', 'CV']], 'left', on='size')
    
    # CV vs input size curve (generated)
    plt.figure()
    sns.lineplot(df, x='input size (nS)', y=r'$g_I$ CV', errorbar='se', color='black')
    sns.scatterplot(df_empirical, x='size', y='CV', color='red')
    
    # mean curves with standard error
    plt.figure()
    sns.lineplot(df, x='input size (nS)', y='firing rate (Hz)', errorbar='se', color='black')
    sns.lineplot(df_empirical_avgs, x='size', y='rate', errorbar='se', linestyle='', err_style='bars', color='red', marker='o')
    
    # scatter plots
    plt.figure()
    sns.scatterplot(df, x='input size (nS)', y='firing rate (Hz)', color='black')
    sns.scatterplot(df_empirical_avgs, x='size', y='rate', color='red')
    
    # print(df_avg)
    # df_avg['input size (nS)'] = dfs[0]['input size (nS)']
    
    # plt.figure()
    # sns.scatterplot(df_empirical, x='size', y='CV')
    # sns.lineplot(df, x='input size (nS)', y=r'$g_I$ CV', errorbar='se', color='black')
    
    
    # plt.figure()
    # sns.scatterplot(df, x='input size (nS)', y='firing rate (Hz)')
    # sns.scatterplot(df_empirical, x='size', y='W_WaveAverage')
    # sns.lineplot(df, x='input size (nS)', y='firing rate (Hz)', errorbar='se', color='black')
    
    # rate vs CV
    plt.figure()
    # sns.scatterplot(df, x=r'$g_I$ CV', y='firing rate (Hz)')
    sns.lineplot(df, x=r'$g_I$ CV', y='firing rate (Hz)', errorbar='se', color='black')
    # sns.scatterplot(df_empirical, x='CV', y='W_WaveAverage', color='red')
    # sns.scatterplot(df_empirical_avgs, x='CV', y='rate')
    sns.lineplot(df_empirical_avgs, x='CV', y='rate', errorbar='se', linestyle='', err_style='bars', color='red', marker='o')
    
    
    
    plt.show()
    

def plot_uniforminputs(**kwargs):
    '''
    This gets figures 4e,f,g for the manuscript
    '''
    kwargs_list = []
    default_kwargs = {'target': 'cython',
                      'report': None,
                      'T': 10*br.second,
                      'N_DCN': 1,
                      # 'V_L': 0*br.mV,
                      'N_extE_DCN': 30000, #6500,
                      'mean_PC_ISI': 1/(100*br.Hz),
                      'total_conductance': 200*br.nS,
                      'empirical_PC': 2,
                      'parallel': True,
                      }
    for key in default_kwargs:
        if key not in kwargs: kwargs[key] = default_kwargs[key]
    if 'out' not in kwargs:
        netfn = kwargs.get('netfn', 'run_DCN_wave')
        kwargs['out'] = {
            'gI_mean': {'run_DCN_wave': "np.mean(gI.values)",
                       'run_DCN_conductance': "np.mean(net['st_DCN'].gI)"}[netfn],
            'gI_std': {'run_DCN_wave': "np.std(gI.values)",
                       'run_DCN_conductance': "np.std(net['st_DCN'].gI)"}[netfn],
            'mean_DCN_rate': "np.mean([len(tr) for tr in net['sp_DCN'].spike_trains().values()])",
            }
    for N_inputs in kwargs.get('input_range', range(5,200)):
        kwargs_list.append({**kwargs,
                            'N_inputs': N_inputs,
                            })
    parallel = kwargs['parallel']
    with Pool(16) if parallel else analysis.yieldexpr(None) as p:
        out_list = list(
            tqdm(
            analysis.starstar(
            partial(p.imap, chunksize=1) if parallel else map,
            run_uniforminputs,
            kwargs_list),
            total=len(kwargs_list))
            )
    df_data = {'input size (nS)': [],
               r'$g_I$ CV': [],
               'firing rate (Hz)': [],
               }
    for out, kwargs_sim in zip(out_list, kwargs_list):
        df_data['input size (nS)'].append(kwargs_sim['total_conductance']/kwargs_sim['N_inputs']/br.nS if kwargs_sim['N_inputs'] else 0)
        df_data[r'$g_I$ CV'].append(out["gI_std"]/out["gI_mean"])
        df_data['firing rate (Hz)'].append(out["mean_DCN_rate"]/kwargs_sim['T']/br.Hz)
    df = pd.DataFrame(df_data)
    if kwargs.get('return_df', False): return df
    df_empirical = pd.read_csv('data/size++.txt')
    sns.lineplot(df, x='input size (nS)', y=r'$g_I$ CV')
    sns.scatterplot(df_empirical, x='size', y='CV')
    plt.figure()
    sns.lineplot(df, x='input size (nS)', y='firing rate (Hz)')
    sns.scatterplot(df_empirical, x='size', y='rate')
    plt.figure()
    sns.lineplot(df, x=r'$g_I$ CV', y='firing rate (Hz)')
    sns.scatterplot(df_empirical, x='CV', y='rate')
    plt.show()


def fit_constE(**kwargs):
    N_inputs_list = [5,10,20,40,80, None]
    kwargs_list = []
    default_kwargs = {
        'T': 1*br.second,
        'N_DCN': 1,
        'total_conductance': 200*br.nS,
        'empirical_PC': 2,
        'load_gI_array': 'data/fig4_gI_array_nS',
        'parallel': True,
        'target_spontaneous_rate': 50*br.Hz,
        'plot': True,
        'out': {
            'gI_mean': "np.mean(gI.values)",
            'gI_std': "np.std(gI.values)",
            'mean_DCN_rate': "np.mean([len(tr) for tr in net['sp_DCN'].spike_trains().values()])",
            },
        }
    for key in default_kwargs:
        if key not in kwargs: kwargs[key] = default_kwargs[key]
    
    
    
    for N_inputs in N_inputs_list:
        kwargs_list.append({**kwargs,
                            'N_inputs': N_inputs,
                            })
    # add the spontaneous activity simulation
    kwargs_list.append({**kwargs_list[0],
                        'gE': 0,
                        'gI': 0})
    xs = []
    parallel = kwargs['parallel']
    
    def callback(x):
        xs.append(x[:])
        print(f'iter {len(xs)}: {x}')
        
    with Pool(len(N_inputs_list)+1) if parallel else analysis.yieldexpr(None) as p:
        try:
            x_fit = scipy.optimize.direct(lossfn_constE,
                                          ((-60,0), # V_L
                                           (0, 100), # gL
                                           (-75, -45), # Vr
                                           (50, 150), # C_m
                                           (-55, -35), # theta
                                           (0, 100), # gE
                                           ),
                                          args=(p, kwargs_list),#, dfs, xs, losses),
                                          maxiter=kwargs.get('maxiter', 50),
                                          callback=callback).x
        except Exception as e:
            x_fit = xs[-1]
            print(f'outer: {e}')
        print(kwargs)
        if kwargs.get('plot', False):
            import matplotlib.animation as animation
            fig, ax = plt.subplots()
            losses, dfs = zip(*[lossfn_constE(x, p,
                                                     kwargs_list,
                                                     return_df=True) for x in xs])
            ani = animation.FuncAnimation(plt.gcf(), plot_df,
                                          frames=zip(range(len(xs)), xs, losses, dfs),
                                          fargs=(kwargs,),
                                         )
            writer = animation.PillowWriter(fps=10)
            x_str = '_'.join([f'{x:.1f}' for x in x_fit])
            ani.save(f"{PLOTNAME}_{x_str}_niters_{len(xs)}.gif", writer=writer)
    return x_fit

def lossfn_constE(x, p, kwargs_list, return_df=False):
    for kwargs in kwargs_list:
        kwargs['V_L'] = x[0]*br.mV
        kwargs['gL'] = x[1]*br.nS
        kwargs['Vr'] = x[2]*br.mV
        kwargs['C_m'] = x[3]*br.pF
        kwargs['theta'] = x[4]*br.mV
        kwargs['gE'] = x[5]*br.nS
        
    if (kwargs['Vr'] > kwargs['theta']
        or kwargs['V_L'] < kwargs['theta']): 
        return np.inf
    
    
    
    
    out_list = list(
        tqdm(
        analysis.starstar(
        partial(p.imap, chunksize=1) if p else map,
        run_uniforminputs,
        kwargs_list),
        total=len(kwargs_list),
        disable=True
        )
        )
    df_data = {'input size (nS)': [],
               r'$g_I$ CV': [],
               'firing rate (Hz)': [],
               }
    for out, kwargs_sim in zip(out_list[:-1], kwargs_list[:-1]):
        df_data['input size (nS)'].append(kwargs_sim['total_conductance']/kwargs_sim['N_inputs']/br.nS if kwargs_sim['N_inputs'] else 0)
        df_data[r'$g_I$ CV'].append(out["gI_std"]/out["gI_mean"])
        df_data['firing rate (Hz)'].append(out["mean_DCN_rate"]/kwargs_sim['T']/br.Hz)
    df = pd.DataFrame(df_data)
    df_rates = df['firing rate (Hz)']
    df_cell = DF_AVERAGES
    df_cell_rates = df_cell['rate']
    loss = 0
    for size in df_cell['size']:
        loss += abs( df_rates[df['input size (nS)']==size].iloc[0] - df_cell_rates[df_cell['size']==size].iloc[0] )
    spontaneous_rate = out_list[-1]["mean_DCN_rate"]/kwargs_list[-1]['T']
    print(spontaneous_rate)
    loss += 10*abs((spontaneous_rate - kwargs['target_spontaneous_rate'])/br.Hz)
    print(f'loss={loss}, x={x}')
    return (loss, df) if return_df else loss

def fit_uniforminputs(**kwargs):
    N_inputs_list = [5,10,20,40,80, None]
    kwargs_list = []
    default_kwargs = {'target': 'cython',
                      'report': None,
                      'T': 1*br.second,
                      'N_DCN': 1,
                      'total_conductance': 200*br.nS,
                      'empirical_PC': 2,
                      'load_gI_array': 'data/fig4_gI_array_nS',
                      'load_gE_array': 'data/fig4_gE_array_nS',
                      'parallel': True,
                      'cellname': '0830_cell_1',
                      'trial': 1,
                      'plot': True,
                      'tau_rp': 1*br.ms,
                      }
    for key in default_kwargs:
        if key not in kwargs: kwargs[key] = default_kwargs[key]
    if 'out' not in kwargs:
        netfn = kwargs.get('netfn', 'run_DCN_wave')
        kwargs['out'] = {
            'gI_mean': {'run_DCN_wave': "np.mean(gI.values)",
                       'run_DCN_conductance': "np.mean(net['st_DCN'].gI)"}[netfn],
            'gI_std': {'run_DCN_wave': "np.std(gI.values)",
                       'run_DCN_conductance': "np.std(net['st_DCN'].gI)"}[netfn],
            'mean_DCN_rate': "np.mean([len(tr) for tr in net['sp_DCN'].spike_trains().values()])",
            }
    
    params = DF_FIG4PARAMS[DF_FIG4PARAMS['name']==kwargs['cellname']].iloc[0]
    # kwargs['C_m'] = params['C capacitance (pF)']*br.pF
    kwargs['N_extE_DCN'] = EMPIRICAL_N_extE_DCN[f"E{params['E conductance (nS)']}_times_A"]
    # kwargs['theta'] = params['firing threshold (mV)']*br.mV
    
    for N_inputs in N_inputs_list:
        kwargs_list.append({**kwargs,
                            'N_inputs': N_inputs,
                            })
    xs = []
    parallel = kwargs['parallel']
    
    def callback(x):
        xs.append(x[:])
        print(f'iter {len(xs)}: {x}')
        
    with Pool(len(N_inputs_list)) if parallel else analysis.yieldexpr(None) as p:
        try:
            x_fit = scipy.optimize.direct(lossfn_uniforminputs,
                                          ((-60,0), # V_L
                                           (0, 100), # gL
                                           (-80, -40), # Vr
                                           (0, 200), # C_m
                                           (-60, -20) # theta
                                           ),
                                          args=(p, kwargs_list),
                                          maxiter=kwargs.get('maxiter', 50),
                                          callback=callback).x
        except Exception as e:
            x_fit = xs[-1]
            print(f'outer: {e}')
        print(kwargs)
        if kwargs.get('plot', False):
            import matplotlib.animation as animation
            fig, ax = plt.subplots()
            losses, dfs = zip(*[lossfn_uniforminputs(x, p,
                                                     kwargs_list,
                                                     return_df=True) for x in xs])
            ani = animation.FuncAnimation(plt.gcf(), plot_df,
                                          frames=zip(range(len(xs)), xs, losses, dfs),
                                          fargs=(kwargs,),
                                         )
            writer = animation.PillowWriter(fps=10)
            x_str = '_'.join([f'{x:.1f}' for x in x_fit])
            ani.save(f"{PLOTNAME}{kwargs['cellname']}_trial_{kwargs['trial']}_{x_str}_niters_{len(xs)}.gif", writer=writer)
    return x_fit

def plot_df(frame, kwargs):
    plt.clf()
    fit_i, x_fit, loss, df = frame
    if 'cellname' not in kwargs:
        sns.lineplot(df, x='input size (nS)', y='firing rate (Hz)')
        plt.scatter(DF_AVERAGES['size'], DF_AVERAGES['rate'])
        x_str = ','.join([f'{x:.1f}' for x in x_fit])
        plt.title(f'iter {fit_i}: {x_str}, loss={loss:.4f}')
        return
    cellname = kwargs['cellname']
    trial = kwargs['trial']
    df_cell = DF_CELLRECS[cellname]
    plt.plot(df['input size (nS)'], df['firing rate (Hz)'])
    for k in df_cell.keys():
        if k != 'size': 
            plt.plot(df_cell['size'], df_cell[k], color='black' if k == f'trial {trial}' else 'grey')
    plt.scatter(df_cell['size'], df_cell['trial_avg'], color='grey')
    x_str = ','.join([f'{x:.1f}' for x in x_fit])
    plt.title(f'iter {fit_i}: {x_str}, loss={loss:.4f}')



def lossfn_uniforminputs(x, p, kwargs_list, return_df=False):
    for kwargs in kwargs_list:
        kwargs['V_L'] = x[0]*br.mV
        kwargs['gL'] = x[1]*br.nS
        kwargs['Vr'] = x[2]*br.mV
        kwargs['C_m'] = x[3]*br.pF
        kwargs['theta'] = x[4]*br.mV
    
    if (kwargs['Vr'] > kwargs['theta']
        or kwargs['V_L'] < kwargs['theta']): 
        return np.inf
        
    out_list = list(
        tqdm(
        analysis.starstar(
        partial(p.imap, chunksize=1) if p else map,
        run_uniforminputs,
        kwargs_list),
        total=len(kwargs_list),
        disable=True
        )
        )
    df_data = {'input size (nS)': [],
               r'$g_I$ CV': [],
               'firing rate (Hz)': [],
               }
    for out, kwargs_sim in zip(out_list, kwargs_list):
        df_data['input size (nS)'].append(kwargs_sim['total_conductance']/kwargs_sim['N_inputs']/br.nS if kwargs_sim['N_inputs'] else 0)
        df_data[r'$g_I$ CV'].append(out["gI_std"]/out["gI_mean"])
        df_data['firing rate (Hz)'].append(out["mean_DCN_rate"]/kwargs_sim['T']/br.Hz)
    df = pd.DataFrame(df_data)
    df_rates = df['firing rate (Hz)']
    cellname = kwargs_list[0]['cellname']
    trial = kwargs_list[0]['trial']
    df_cell = DF_CELLRECS[cellname]
    df_cell_rates = df_cell[f"trial {trial}" if trial else 'trial_avg']
    loss = 0
    for size in df_cell['size']:
        loss += abs( df_rates[df['input size (nS)']==size].iloc[0] - df_cell_rates[df_cell['size']==size].iloc[0] )
    print(f'loss={loss}, x={x}')
    return (loss, df) if return_df else loss
    

     
def run_uniforminputs(**kwargs):
    try:
        N_inputs = kwargs['N_inputs']
        # print(N_inputs)
        if not N_inputs:
            # use constant conductance wave
            if 'gI' not in kwargs: kwargs['gI'] = 56*br.nS
            return network.run_DCN_wave(**kwargs)
        if 'empirical_PC' in kwargs:
            empiricalPC_spiketimes_sec = [x[~np.isnan(x)] for x in
                                      pd.read_csv('data/PC in vivo data_latest.csv').values.T]
            # empirical_rates_Hz = [len(x)/(x[-1]-x[0]) for x in empiricalPC_spiketimes_sec]
            # print(empirical_rates_Hz)
            ISI_list = np.diff(empiricalPC_spiketimes_sec[kwargs['empirical_PC']-1])
            kwargs['PC_indices'], kwargs['PC_times'] = analysis.generate_PC_spikes(ISI_list, br.second, range(kwargs['N_DCN']*N_inputs), kwargs['T'])
        total_conductance = kwargs['total_conductance']
        kwargs['input_sizes'] = [total_conductance/N_inputs for _ in range(N_inputs)]
        return fig2.run_differentsizeinputs(**kwargs)
    except KeyboardInterrupt as e:
        print(f'inner: {e}')
        return
    # net = fig2.run_differentsizeinputs(**kwargs)
    # out = kwargs.get('out', 'net')
    # if out: return eval(out) if isinstance(out, str) else dict(zip(out, map(eval, out)))






# full input distribution
import fig6

G_nS = fig2.G_nS
FINAL_PARAMS = fig2.FINAL_PARAMS
UNIFORM_PARAMS = fig6.UNIFORM_PARAMS

def run_fulldist(**kwargs):
    kwargs = {**{
        'out': {
            'DCN_rate_Hz': "[len(tr) for tr in net['sp_DCN'].spike_trains().values()]/net.t/br.Hz",
            'gI_mean_nS': "np.mean(gI.values.T*1e9, axis=1)",
            'gI_std_nS': "np.std(gI.values.T*1e9, axis=1)",
            },
        'report': None,
        'record': False,
        'T': 10*br.second,
        }, 
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs}
    
    totalconductance_nS = kwargs.get('totalconductance', 200*br.nS)/br.nS
    kwargs_list = []
    for seed in range(kwargs.get('nseeds', 100)):
        input_sizes_nS = []
        with analysis.seed(seed):
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
        input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
        input_sizes_nS.sort()
        kwargs_list.append({
            **kwargs,
            'input_sizes': input_sizes_nS*br.nS,
            'seed': seed,
            'uniform': False,
            })
    
    for N_inputs in range(1, 201):
        size_nS = totalconductance_nS/N_inputs
        input_sizes_nS = [size_nS for _ in range(N_inputs)]
        kwargs_list.append({
            **kwargs,
            'input_sizes': input_sizes_nS*br.nS,
            'seed': None,
            'uniform': True,
            })
    
    with Pool(16) as p:
        sim_list = list(tqdm(analysis.starstar(p.imap, fig2.run_differentsizeinputs, kwargs_list),
                             total=len(kwargs_list)))
    
    df_data = {key: [] for key in [
        'seed',
        'uniform',
        'output rate (Hz)',
        '$g_I$ mean (nS)',
        '$g_I$ std (nS)',
        '$g_I$ CV',
        ]}
    
    for sim, kwargs_sim in zip(sim_list, kwargs_list):
        df_data['seed'].append(kwargs_sim['seed'])
        df_data['uniform'].append(kwargs_sim['uniform'])
        df_data['output rate (Hz)'].append(sim['DCN_rate_Hz'][0])
        df_data['$g_I$ mean (nS)'].append(sim['gI_mean_nS'][0])
        df_data['$g_I$ std (nS)'].append(sim['gI_std_nS'][0])
        df_data['$g_I$ CV'].append(sim['gI_std_nS'][0]/sim['gI_mean_nS'][0])
        
    df = pd.DataFrame(df_data)
    df.to_csv(f'{PLOTNAME}_fig4_fulldist_uniformparams.csv')

def get_conductances(**kwargs):
    kwargs = {**{
        'out': {
            'gI_nS': "gI.values.T[0]*1e9",
            },
        'report': None,
        'record': False,
        'T': 10*br.second,
        }, 
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs}
    
    totalconductance_nS = kwargs.get('totalconductance', 200*br.nS)/br.nS
    kwargs_list = []
    for seed in range(kwargs.get('nseeds', 10)):
        input_sizes_nS = []
        with analysis.seed(seed):
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
        input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
        input_sizes_nS.sort()
        kwargs_list.append({
            **kwargs,
            'input_sizes': input_sizes_nS*br.nS,
            'seed': seed,
            'uniform': False,
            })
    
    # for N_inputs in range(1, 201):
    #     size_nS = totalconductance_nS/N_inputs
    #     input_sizes_nS = [size_nS for _ in range(N_inputs)]
    #     kwargs_list.append({
    #         **kwargs,
    #         'input_sizes': input_sizes_nS*br.nS,
    #         'seed': None,
    #         'uniform': True,
    #         })
    
    with Pool(16) as p:
        sim_list = list(tqdm(analysis.starstar(p.imap, fig2.run_differentsizeinputs, kwargs_list),
                             total=len(kwargs_list)))
    
    df_data = {key: [] for key in [
        'seed',
        'time (s)',
        '$g_I$ (nS)',
        ]}
    
    df_data_sizes = {key: [] for key in [
        'seed',
        'size (nS)',
        ]}
    
    for sim, kwargs_sim in zip(sim_list, kwargs_list):
        len_gI = len(sim['gI_nS'])
        # for i, gI in enumerate(sim['gI_nS']):
        #     df_data['seed'].append(kwargs_sim['seed'])
        #     df_data['time (s)'].append(kwargs_sim['T']*i/len_gI)
        #     df_data['$g_I$ (nS)'].append(gI)
        for size in kwargs_sim['input_sizes']:
            df_data_sizes['seed'].append(kwargs_sim['seed'])
            df_data_sizes['size (nS)'].append(size/br.nS)
        
    df = pd.DataFrame(df_data)
    df.to_csv(f'{PLOTNAME}_fig4_conductances_uniformparams.csv')
    
    df_sizes = pd.DataFrame(df_data_sizes)
    df_sizes.to_csv(f'{PLOTNAME}_fig4_sizes_uniformparams.csv')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        result = eval(' '.join(sys.argv[1:]))