import numpy as np
import scipy
import brian2.only as br

import analysis

import os

IPSC_timestep = 20*br.us

IPSC = np.genfromtxt('data/PC_IPSC_20ms.csv')[1:]  # 20 us timestep
IPSC = IPSC/max(IPSC)

EPSC = np.genfromtxt('data/MF_EPSC_20ms.csv')[1:]
EPSC = EPSC/max(EPSC)

def run_DCN_wave(**kwargs):
    
    target = kwargs.get('target', 'cython')
    if target == 'numpy':
        br.set_device('runtime')
        br.prefs.codegen.target = 'numpy'
    elif target == 'cython':  # can only run 640 at a time with cython
        br.set_device('runtime')
        br.prefs.codegen.target = 'cython'
    else:
        br.set_device('cpp_standalone', directory=f'standalone/{os.getpid()}')
        br.device.reinit()
        br.device.activate(directory=f'standalone/{os.getpid()}')
    
    T = kwargs.get('T', 1*br.second)
    
    if 'PC_indices' not in kwargs and 'gI' not in kwargs:
        N_PC = kwargs.get('N_PC', 1)
        PC_indices, PC_times = analysis.fitted_PC_spikes(range(N_PC), T, mean_PC_ISI=kwargs.get('mean_PC_ISI', 1/(80*br.Hz)))
    else:
        PC_indices = kwargs.get('PC_indices', [0])
        PC_times = kwargs.get('PC_times', [0]*br.second)
        N_PC = max(PC_indices) + 1
    PC = br.SpikeGeneratorGroup(N_PC, PC_indices, PC_times, name='PC')
    
    tau_rp = kwargs.get('tau_rp', 2*br.ms)#'(2+0*rand())*ms')
    theta = kwargs.get('theta', -50*br.mV)
    Vr = kwargs.get('Vr', -60*br.mV)  # check -50 to -60
    gL = kwargs.get('gL', 8.8*br.nS)
    C_m = kwargs.get('C_m', 70*br.pF)
    V_L = kwargs.get('V_L', -40*br.mV)  # check: -40 or 0
    V_E = 0*br.mV
    V_I = -75*br.mV
    N_DCN = kwargs.get('N_DCN', 1)
    N_gI_timesteps = int(PC_times[-1]/IPSC_timestep) + 1  # PC_times is sorted
    N_timesteps = int(T/IPSC_timestep) + 1
    
    if kwargs.get('gI', False):
        gI = br.TimedArray([[kwargs['gI']]], T)
    else:
        # generate the conductance wave
        # gI_array = np.zeros([N_DCN, N_gI_timesteps]) * br.nS
        connect = kwargs.get('connect', {'i': [], 'j': []})
        gI_array_loaded = False
        load_gI_array = kwargs.get('load_gI_array', False)
        if load_gI_array:
            gI_array_name = f"{load_gI_array}_{N_PC}_{N_DCN}_T_{T/br.second}.npy"
            try:
                gI_array_nS = np.load(gI_array_name)
                gI_array_loaded = True
            except FileNotFoundError:
                pass
        if not gI_array_loaded:
            w_list = np.array(
                kwargs.get('w',
                           [1*br.nS for _ in range(len(connect['i']))]
                           )
                )
            # print([N_DCN, N_gI_timesteps])
            weightedtimes = np.zeros([N_DCN, N_gI_timesteps])
            PC_timesteps = np.array(PC_times/IPSC_timestep, dtype=int)
            for i_PC, j_DCN, w in zip(connect['i'], connect['j'], w_list):
                weightedtimes[j_DCN, PC_timesteps[PC_indices==i_PC]] += (w/br.nS)
            gI_array_nS = np.array(
                [np.convolve(weightedtimes[j_DCN], IPSC, 'full')[:N_timesteps]
                 for j_DCN in range(N_DCN)]
                )
            if kwargs.get('complete_gI', False):
                weightedtimes_complete = np.zeros([N_DCN, len(w_list), N_gI_timesteps])
                for i_PC, j_DCN, w, idx in zip(connect['i'], connect['j'], w_list, range(len(w_list))):
                    weightedtimes_complete[j_DCN, idx, PC_timesteps[PC_indices==i_PC]] += (w/br.nS)
                gI_array_complete_nS = np.array(
                    [[scipy.signal.convolve(weightedtimes_complete[j_DCN, idx], IPSC, 'full')[:N_timesteps]
                      for idx in range(len(w_list))] for j_DCN in range(N_DCN)]
                    )
        if load_gI_array and not gI_array_loaded:
            np.save(gI_array_name, gI_array_nS)
        gI_array = gI_array_nS * br.nS
        gI = br.TimedArray(gI_array.T, dt=IPSC_timestep)
    # print(f'tau_I: {sum(IPSC>(1/np.e))*IPSC_timestep}')
    
    if kwargs.get('gE', False):
        gE = br.TimedArray([kwargs['gE']], T)
    else:
        N_extE_DCN = kwargs.get('N_extE_DCN', 0)
        gE_array_loaded = False
        load_gE_array = kwargs.get('load_gE_array', False)
        if load_gE_array:
            gE_array_name = f"{load_gE_array}_{N_extE_DCN}_T_{T/br.second}.npy"
            try:
                gE_array_nS = np.load(gE_array_name)
                gE_array_loaded = True
            except FileNotFoundError:
                pass
        if not gE_array_loaded:
            gE_array_nS = 0.4 * scipy.signal.convolve(
                np.random.poisson(N_extE_DCN*(IPSC_timestep/br.second), N_timesteps),
                EPSC,
                'full'
                )[:N_timesteps]
        if load_gE_array and not gE_array_loaded:
            np.save(gE_array_name, gE_array_nS)
        gE_array = gE_array_nS * br.nS
        gE = br.TimedArray(gE_array, dt=IPSC_timestep)
    # print(f'tau_E: {sum(EPSC>(1/np.e))*IPSC_timestep}')
    
    DCN = br.NeuronGroup(N_DCN,
                         '''
                         dv/dt = (gL*(V_L-v) + gE(t)*(V_E-v) + gI(t,i)*(V_I-v)) / C_m : volt (unless refractory)
                         ''',
                         threshold='v>theta',
                         reset='v=Vr',
                         refractory=tau_rp,
                         method='euler',
                         name='DCN')
    DCN.v = Vr
    sp_DCN = br.SpikeMonitor(DCN, name='sp_DCN')
    st_DCN = br.StateMonitor(DCN, kwargs.get('variables', ['v',
                                                           # 'gE',
                                                           # 'gI',
                                                           ]), record=kwargs.get('record', True), name='st_DCN')
    sp_PC = br.SpikeMonitor(PC, name='sp_PC')

    net = br.Network(PC, DCN,
                     # extE_DCN,
                     sp_DCN,
                     sp_PC,
                     st_DCN)
    net.run(T, report=kwargs.get('report', None)) # 'stdout', 'stderr' or None
    
    if 'out' in kwargs:
        result = {}
        for out_key, out_value in kwargs['out'].items(): result[out_key] = eval(out_value)
        return result
    return net
    # This should work but does not (python bug?):
    # return {out_key: eval(out_value) for out_key, out_value in kwargs['out'].items()} if 'out' in kwargs else net
