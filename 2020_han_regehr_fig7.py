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
import itertools
import pickle

mpl.rcParams["axes.labelpad"] = 0
mpl.rcParams["lines.color"] = "black"

PARAMS = {
    "mean_PC_ISI": 1 / (80 * br.Hz),
    "C_m": 70 * br.pF,
    "N_extE_DCN": 25000,
    "gL": 20 * br.nS,
    "V_L": -49.9 * br.mV,
    "tau_rp": 1 * br.ms,
}

G_nS = fig2.G_nS

PLOTNAME = "figs/20231004"


def run_shuting_pauses_worker(**kwargs):
    """Run the pause simulation for a cell mimicking Shuting's experiment."""
    with analysis.seed(kwargs.get("seed")):
        mean_PC_ISI = kwargs.pop("mean_PC_ISI", 1 / (80 * br.Hz))
        PC_spikes_transient = 10 * mean_PC_ISI
        PC_indices, PC_times = analysis.fitted_PC_spikes(
            range(len(kwargs["input_sizes"])),
            kwargs["T"] + PC_spikes_transient,
            mean_PC_ISI=mean_PC_ISI,
        )
        PC_times = PC_times - PC_spikes_transient
        # Generate the PC spike times with random pause choices
        trial_gap = kwargs.pop("trial_gap")
        pause_length = kwargs.pop("pause_length")
        pause_indices_dict = kwargs.pop("pause_indices_dict")
        kwargs["pause_neurons_list"] = []
        pause_mask = PC_times < 0
        # pause_t = kwargs.pop("transient")

        pause_indices_list = list(pause_indices_dict)

        for pause_t_ms in kwargs.pop("pause_times_ms"):
            # while pause_t < kwargs["T"]:
            # pause_neurons = np.random.choice(pause_indices_list)
            pause_neurons = pause_indices_list[
                np.random.randint(len(pause_indices_list))
            ]
            kwargs["pause_neurons_list"].append(pause_neurons)
            pause_mask |= (
                (PC_times / br.ms > pause_t_ms)
                & (PC_times / br.ms < pause_t_ms + pause_length / br.ms)
                & (np.isin(PC_indices, pause_indices_dict[pause_neurons]))
            )
            # pause_t += trial_gap

        # print(kwargs['seed'], kwargs['pause_neurons_list'])

        kwargs["PC_indices"] = PC_indices[~pause_mask]
        kwargs["PC_times"] = PC_times[~pause_mask]

        # if kwargs.get("seed") == 0:
        #     pd.DataFrame(
        #         {
        #             "PC_indices": kwargs["PC_indices"],
        #             "PC times (ms)": kwargs["PC_times"]/br.ms,
        #         }
        #     ).to_csv(kwargs['shuting_plotname']+'_seed_0_PC_raster.csv')
        # return

        return fig2.run_differentsizeinputs(**kwargs)


def run_shuting_pauses(**kwargs):
    """

    Batch:
    ```bat
        for %g in (20 50 100) do for %p in ('uniform' '') do python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, trial_gap=%g*br.ms, prefix=%p)
    ```

    For 32 inputs:
    ```bat
        for %g in (20 50 100) do python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, trial_gap=%g*br.ms, prefix='32inputs', seed=68)
    ```

    For 24 inputs with less excitation:
    ```bat
        for %g in (20 50 100) do for %p in ('uniform' '') do python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, trial_gap=%g*br.ms, prefix=%p+'_exwave2', N_extE_DCN=20000)
    ```

    For the dynamic clamp experiments (20230905):
    ```bat
        python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, trial_gap=20*br.ms, prefix='shuting_24inputs', N_extE_DCN=23000, tau_rp=1*br.ms) & python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, trial_gap=20*br.ms, prefix='shuting_uniform', tau_rp=1*br.ms)
    ```

    For the dynamic clamp experiments (20230923):
    ```bat
        python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, prefix='shuting_24inputs', N_extE_DCN=23000) & python 2020_han_regehr_fig7.py run_shuting_pauses(T=100*br.second, prefix='shuting_uniform')
    ```

    ```bat
        for %g in (20 50 100) do python 2020_han_regehr_fig7.py run_shuting_pauses(trial_gap=%g*br.ms, T=100*br.second, prefix='shuting_24inputs', N_extE_DCN=23000) & python 2020_han_regehr_fig7.py run_shuting_pauses(trial_gap=%g*br.ms, T=100*br.second, prefix='shuting_uniform')
    ```

    many seeds (dists with 24 inputs)
    ```bat
        for %s in (20,  27,  45,  64,  71, 113, 141, 145, 162, 167) do python 2020_han_regehr_fig7.py run_shuting_pauses(seed=%s, T=100*br.second, nreps=160, prefix='random_seed_%s', N_extE_DCN=23000)
    ```
    To save the rates over many seeds in a single file:
    ```python
        pd.concat([pd.read_csv(f"figs/20230928_gap_20ms_random_seed_{seed}_rates.csv", index_col=0).assign(seed=seed) for seed in (20,  27,  45,  64,  71, 113, 141, 145, 162, 167)], ignore_index=True).to_csv('figs/20230928_gap_20ms_random_24inputseeds_rates.csv')
    ```
    """
    kwargs = {
        **{
            "report": None,
            # 'record': False,
            # "target": "numpy",
            "T": 10 * br.second,  # can do 100s
            "transient": 250 * br.ms,
            "nreps": 16,
            "totalconductance_nS": 200,
            "trial_gap": 20 * br.ms,
            "pause_length": 2 * br.ms,
            "seed": 20,  # seeds with 24 inputs: [20, 27, 45, 64, 71]
            "out": {
                "PC_trs": "net['sp_PC'].spike_trains()",
                "DCN_trs": "net['sp_DCN'].spike_trains()",
                "DCN_v": 'net["st_DCN"].v',
                "t": "net.t",
                "pause_neurons_list": "kwargs['pause_neurons_list']",
                # "gI_nS": "gI_array_nS[0]",
            },
        },
        **PARAMS,
        # **fig2.FINAL_PARAMS,
        **kwargs,
    }

    save_gI = "gI_nS" in kwargs["out"]

    plotname = f"{PLOTNAME}_gap_{kwargs['trial_gap']/br.ms:.0f}ms"
    prefix = kwargs.pop("prefix", "")
    if prefix:
        plotname += f"_{prefix}"
    kwargs["shuting_plotname"] = plotname

    plot_raster = True

    totalconductance_nS = kwargs.pop("totalconductance_nS")
    if "uniform" in prefix:
        if 'full' in prefix: plot_raster = False
        N_inputs = kwargs.pop("N_inputs", 40)
        kwargs["input_sizes"] = [
            totalconductance_nS / N_inputs * br.nS for _ in range(N_inputs)
        ]
        kwargs["pause_indices_dict"] = (
            {
                **{r"0%": tuple()},
                **{
                    rf"{int(100*pause_fraction)}%": tuple(
                        range(int(0 * N_inputs), int((pause_fraction) * N_inputs))
                    )
                    for pause_fraction in (0.25, 0.5)
                },
                **{r"100%": tuple(range(N_inputs))},
            }
            if "full" not in prefix
            else {
                **{r"0%": tuple()},
                **{
                    pause_neurons: pause_neurons
                    for pause_neurons in [range(n) for n in range(N_inputs)]
                },
                **{r"100%": tuple(range(N_inputs))},
            }
        )
        input_sizes_nS = kwargs["input_sizes"] / br.nS
    elif "random" in prefix:
        plot_raster = False
        with analysis.seed(kwargs["seed"]):
            input_sizes_nS = []
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
            input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
            input_sizes_nS.sort()
        kwargs["input_sizes"] = input_sizes_nS * br.nS
        N_inputs = len(input_sizes_nS)
        # no pause, 1 input paused, 100% paused, all but 1 input paused
        kwargs["pause_indices_dict"] = {
            **{r"0%": tuple()},
            **{
                pause_neurons: pause_neurons
                for pause_neurons in [
                    tuple(
                        sorted(
                            np.random.choice(
                                range(N_inputs),
                                size=np.random.randint(N_inputs - 1) + 1,
                                replace=False,
                            )
                        )
                    )
                    for _ in range(100)
                ]
            },
            **{r"100%": tuple(range(N_inputs))},
        }
    else:
        # shuting's experiment with variable-size inputs
        with analysis.seed(kwargs["seed"]):
            input_sizes_nS = []
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
            input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
            input_sizes_nS.sort()
        kwargs["input_sizes"] = input_sizes_nS * br.nS
        N_inputs = len(input_sizes_nS)
        # 0%, 25%, 50%, 100% of the biggest, medium and smallest inputs
        kwargs["pause_indices_dict"] = {
            **{r"0%": tuple()},
            **{
                rf"{int(100*pause_fraction)}% {size_desc}": tuple(
                    range(
                        int(offset * N_inputs),
                        int((pause_fraction + offset) * N_inputs),
                    )
                )
                for pause_fraction in (0.25, 0.5)
                for size_desc, offset in zip(
                    ["small", "medium", "large"],
                    [0, 0.5 - pause_fraction / 2, 1 - pause_fraction],
                )
            },
            **{r"100%": tuple(range(N_inputs))},
        }
    # print(kwargs["pause_indices_dict"])

    # save input sizes, pause_indices_dict, (pause_neurons_list for each cell)
    pd.Series(input_sizes_nS, name="input sizes (nS)").to_csv(
        f"{plotname}_input_sizes_nS.csv"
    )
    np.savetxt(
        f"{plotname}_pause_indices.txt",
        [f"{k}: {v}" for k, v in kwargs["pause_indices_dict"].items()],
        fmt="%s",
    )

    pause_times_ms = np.arange(
        kwargs["transient"] / br.ms, kwargs["T"] / br.ms, kwargs["trial_gap"] / br.ms
    )
    kwargs["pause_times_ms"] = pause_times_ms

    nreps = kwargs.pop("nreps")
    with Pool(16, initializer=np.random.seed) as p:
        sims = list(
            tqdm(
                analysis.starstar(
                    p.imap,
                    run_shuting_pauses_worker,
                    [
                        {
                            **kwargs,
                            **{
                                "seed": rep,
                                # "save_gI_array": f"{plotname}_gI_{rep}.csv",
                                # "save_gE_array": f"{plotname}_gE_{rep}.csv",
                            },
                        }
                        for rep in range(nreps)
                    ],
                ),
                total=nreps,
            )
        )

    rasters_dict = {pause_neurons: [] for pause_neurons in kwargs["pause_indices_dict"]}

    # compute the relative raster plot around all pauses for each unique elem in pause_indices_list
    nseeds_df = 3
    pause_neurons_df_dict = {
        # "time_ms": np.arange(250, kwargs["T"] / br.ms, kwargs["trial_gap"] / br.ms)
        "time_ms": pause_times_ms,
    }
    if save_gI:
        gI_nS_df_dict = {
            "time_ms": np.arange(0, kwargs["T"] / br.ms, network.IPSC_timestep / br.ms)
        }
    transient = kwargs.pop("transient")
    for sim_idx, sim in tqdm(enumerate(sims)):
        if sim_idx < nseeds_df:
            pause_neurons_df_dict[f"seed {sim_idx}"] = sim["pause_neurons_list"]
            if save_gI:
                gI_nS_df_dict[f"seed {sim_idx}"] = sim["gI_nS"]
        for pause_neurons_idx, pause_neurons in enumerate(sim["pause_neurons_list"]):
            pause_t = transient + kwargs["trial_gap"] * pause_neurons_idx
            tr_ms = (sim["DCN_trs"][0] - pause_t) / br.ms
            tr_mask = (tr_ms > -5) & (tr_ms < kwargs["trial_gap"]/br.ms)
            tr_ms = tr_ms[tr_mask]
            rasters_dict[pause_neurons].append(tr_ms)
    pd.DataFrame(
        [
            (pause_neurons, trial, t_ms / 1000)
            for pause_neurons, tr_ms_list in rasters_dict.items()
            for trial, tr_ms in enumerate(tr_ms_list)
            for t_ms in tr_ms
        ],
        columns=["pause_neurons", "trial", "t (s)"],
    ).to_csv(f"{plotname}_raster.csv")
    pd.DataFrame(pause_neurons_df_dict).to_csv(f"{plotname}_pause_neurons.csv")
    if save_gI:
        pd.DataFrame(gI_nS_df_dict).to_csv(f"{plotname}_gI_nS.csv")

    hist_100, bins_100 = np.histogram(
        np.concatenate(rasters_dict["100%"]),
        np.arange(-5, kwargs["trial_gap"]/br.ms, 0.5),
    )
    maxrate_idx = hist_100.argmax()
    maxrate_label = f"rate_{bins_100[maxrate_idx]}ms_Hz"
    rates_df_dict = {
        k: []
        for k in [
            "size",
            "percentage",
            "total_size_nS",
            "max_rate_Hz",
            "20ms_average_rate_Hz",
            "total_average_rate_Hz",
            maxrate_label,
            "baseline_rate_Hz",
        ]
    }

    nrows = len(kwargs["pause_indices_dict"])
    if plot_raster:
        fig, axs = plt.subplots(nrows, 2, figsize=(5, 2 * nrows))

    for idx, (pause_neurons, tr_ms_list) in tqdm(enumerate(rasters_dict.items())):
        hist, bins = np.histogram(
            np.concatenate([tr_ms for tr_ms in tr_ms_list]),
            np.arange(-5, kwargs["trial_gap"]/br.ms, 0.5),  # 10 * br.defaultclock.dt / br.ms),
        )

        rates_df_dict["size"].append(
            "".join(str(pause_neurons).split(r"%")[1:]).strip()
        )
        rates_df_dict["percentage"].append(
            100 * len(kwargs["pause_indices_dict"][pause_neurons]) / len(input_sizes_nS)
        )
        rates_df_dict["total_size_nS"].append(
            sum(
                [
                    input_sizes_nS[index]
                    for index in kwargs["pause_indices_dict"][pause_neurons]
                ]
            )
        )
        rates_df_dict["max_rate_Hz"].append(
            max(hist) / (len(tr_ms_list) * (bins[1] - bins[0]) / 1000)
        )
        rates_df_dict["20ms_average_rate_Hz"].append(
            hist[(bins[:-1] > 0) & (bins[:-1] < 20)].mean() / (len(tr_ms_list) * (bins[1] - bins[0]) / 1000)
        )
        rates_df_dict["total_average_rate_Hz"].append(
            hist[bins[:-1] > 0].mean() / (len(tr_ms_list) * (bins[1] - bins[0]) / 1000)
        )
        rates_df_dict[maxrate_label].append(
            hist[maxrate_idx] / (len(tr_ms_list) * (bins[1] - bins[0]) / 1000)
        )
        rates_df_dict["baseline_rate_Hz"].append(
            hist[bins[:-1] <= 0].mean() / (len(tr_ms_list) * (bins[1] - bins[0]) / 1000)
        )

        if plot_raster:
            axs[idx, 0].eventplot(tr_ms_list, colors="k")
            axs[idx, 0].set(
                xlabel="time (ms)",
                ylabel="trial",
                title=pause_neurons,
            )
            axs[idx, 1].plot(
                (bins[1:] + bins[:-1]) / 2,
                hist / rates_df_dict["baseline_rate_Hz"][-1],
                "k",
            )
            axs[idx, 1].set(
                xlabel="time (ms)",
                ylabel="rate (norm.)",
                title=pause_neurons,
            )
            analysis.set_lims(
                [min(bins), max(bins)],
                [0, 1.1 * max(hist / rates_df_dict["baseline_rate_Hz"][-1])],
                (1, 1),
                ax=axs[idx, 1],
            )
    if plot_raster:
        fig.tight_layout()
        fig.savefig(f"{plotname}_raster.png")

    rates_df = pd.DataFrame(rates_df_dict)
    print(rates_df)
    rates_df.to_csv(f"{plotname}_rates.csv")
    g = sns.PairGrid(
        rates_df,
        y_vars=[
            "max_rate_Hz",
            "20ms_average_rate_Hz",
            "total_average_rate_Hz",
            maxrate_label,
            "baseline_rate_Hz",
        ],
        x_vars=[
            "percentage",
            "total_size_nS",
        ],
        height=2.5,
        aspect=1.5,
        hue="size" if "uniform" not in prefix else None,
    )
    g.map(sns.scatterplot, linewidth=0)
    g.set(ylim=[0, None])
    g.add_legend()
    g.savefig(f"{plotname}_rates.png")

    # plt.show()


def n_inputs_worker(**kwargs):
    totalconductance_nS = kwargs.get("totalconductance", 200 * br.nS) / br.nS
    seed = kwargs.get("seed", 0)
    with analysis.seed(seed):
        input_sizes_nS = []
        while sum(input_sizes_nS) < totalconductance_nS:
            input_sizes_nS.append(np.random.choice(G_nS))
        input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
    # if len(input_sizes_nS) == 20: print(seed)
    return len(input_sizes_nS)


def n_inputs(**kwargs):
    with Pool(16, initializer=np.random.seed) as p:
        sims = list(
            tqdm(
                analysis.starstar(
                    p.imap,
                    n_inputs_worker,
                    [{"seed": seed} for seed in range(kwargs.get("nseeds", 1000))],
                )
            )
        )
        print(sims)
        print(np.where(np.array(sims) == kwargs.get("length", 24)))
        plt.hist(sims, range(50))
    plt.xlabel("number of inputs")
    plt.ylabel("frequency")
    plt.show()


def run_han_spontaneous(**kwargs):
    """

    (gL=5*br.nS, V_L=-40*br.mV, C_m=70*br.pF): 86 Hz
    (gL=8.8*br.nS, V_L=-40*br.mV, C_m=70*br.pF): >100 Hz
    (gL=3*br.nS, V_L=-40*br.mV, C_m=70*br.pF): 55 Hz
    (gL=3*br.nS, V_L=-40*br.mV, C_m=100*br.pF): 40 Hz

    """
    return run_han_pauses(
        totalconductance_nS=0,
        T=10 * br.second,
        N_extE_DCN=0,
        nreps=16,
        **kwargs,
    )


def run_han_pauses_worker(**kwargs):
    mean_PC_ISI = kwargs.pop("mean_PC_ISI", 1 / (80 * br.Hz))
    PC_spikes_transient = 10 * mean_PC_ISI
    PC_indices, PC_times = analysis.fitted_PC_spikes(
        range(len(kwargs["input_sizes"])),
        kwargs["T"] + PC_spikes_transient,
        mean_PC_ISI=mean_PC_ISI,
    )
    PC_times = PC_times - PC_spikes_transient
    pause_ms = kwargs.pop("pause_ms")
    pause_mask = (  # spikes to remove
        (PC_times / br.ms > pause_ms[0])
        & (PC_times / br.ms < pause_ms[1])
        & (PC_indices < kwargs.pop("pause_frac") * len(kwargs["input_sizes"]))
    ) | (PC_times < 0)
    PC_indices = PC_indices[~pause_mask]
    PC_times = PC_times[~pause_mask]

    kwargs["PC_indices"] = PC_indices
    kwargs["PC_times"] = PC_times

    return fig2.run_differentsizeinputs(**kwargs)


def run_han_pauses(**kwargs):
    """Runs the pause sims in parallel.

    To check the spontaneous firing rates of the cell (with no excitation or inhibition):
    add the parameters `totalconductance_nS=0, T=10*br.second, N_extE_DCN=0, T=10*br.second, nreps=16`
    (average DCN rate for these params: 26 Hz;
        50Hz for gL=1nS;
        34Hz for V_L=-40mV, gL=2.5nS;
        27Hz for V_L=-40mV, gL=1nS, C_m=50pF;

    )
    ```python
        python 2020_han_regehr_fig7.py run_pauses(nreps=16, pause_frac=.5, prefix='test', gL=.5*br.nS, V_L=0*br.mV, N_extE_DCN=0, C_m=100*br.pF, totalconductance_nS=0, T=10*br.second)

        python 2020_han_regehr_fig7.py run_pauses(nreps=16, pause_frac=0, prefix='test', gL=8.8*br.nS, V_L=-40*br.mV, N_extE_DCN=0, C_m=70*br.pF, totalconductance_nS=0, T=10*br.second)

        python 2020_han_regehr_fig7.py run_pauses(pause_frac=.1, prefix='test', gE=17*br.nS, gL=3*br.nS, V_L=-40*br.mV, C_m=100*br.pF, nreps=5000, N_extE_DCN=0, T=350*br.ms, pause_ms=[250,252], xlower_ms=230, mean_PC_ISI=1/(80*br.Hz))
    ```
    """
    kwargs = {
        **{
            "report": None,
            # 'record': False,
            "target": "numpy",
            "T": 50 * br.ms,
            "xlower_ms": 0,
            "pause_ms": [20, 22],
            "pause_frac": 0,
            "nreps": 200,
            "totalconductance_nS": 200,
            # "DCN_v": "-25*rand()*mV - 50*mV",
            "seed": 2,
            "out": {
                "PC_trs": "net['sp_PC'].spike_trains()",
                "DCN_trs": "net['sp_DCN'].spike_trains()",
                "DCN_v": 'net["st_DCN"].v',
                "t": "net.t",
            },
            # 'nbins': 50,
        },
        **PARAMS,
        # **UNIFORM_PARAMS,
        # **fig2.FINAL_PARAMS,
        **kwargs,
    }

    plotname = f"{PLOTNAME}{kwargs.get('prefix', '')}_{kwargs.get('pause_frac')}"

    totalconductance_nS = kwargs.pop("totalconductance_nS")

    pause_ms = kwargs.get("pause_ms")

    xlower_ms = kwargs.pop("xlower_ms")

    N_inputs = kwargs.pop("N_inputs", 40)
    kwargs["input_sizes"] = [
        totalconductance_nS / N_inputs * br.nS for _ in range(N_inputs)
    ]
    # with analysis.seed(kwargs["seed"]):
    #     input_sizes_nS = []
    #     while sum(input_sizes_nS) < totalconductance_nS:
    #         input_sizes_nS.append(np.random.choice(G_nS))
    #     input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
    #     input_sizes_nS.sort()
    # kwargs['input_sizes'] = input_sizes_nS * br.nS

    nreps = kwargs.pop("nreps")

    with Pool(16, initializer=np.random.seed) as p:
        sims = list(
            tqdm(
                analysis.starstar(
                    p.imap,
                    run_han_pauses_worker,
                    [kwargs for _ in range(nreps)],
                ),
                total=nreps,
            )
        )

    # PC raster plot for the first DCN neuron
    plt.figure(figsize=(9 / 3, 4 / 3))
    # PC_trs = np.take_along_axis(np.array([sim['PC_trs'][i]/br.ms for i in range(len(input_sizes_nS))]), np.argsort(input_sizes_nS), axis=0)
    PC_trs = sims[0]["PC_trs"]
    for idx in range(len(kwargs["input_sizes"])):  # len(trs)):
        plt.vlines(PC_trs[idx] / br.ms, idx, idx + 1, "black")
    # plt.xlim([900, 1000])
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of\nPCs")
    analysis.set_lims(
        [xlower_ms, pause_ms[0], kwargs["T"] / br.ms],
        [0, len(kwargs["input_sizes"])],
        minmax=(False, True),
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{plotname}_raster_PC.png")

    DCN_trs = [sim["DCN_trs"][0] for sim in sims]

    # DCN trace
    plt.figure(figsize=(9 / 3, 4 / 3))
    for idx in range(10):  # len(sim['st_DCN'].v)):
        V_mV = sims[idx]["DCN_v"][0] / br.mV
        # print(V_mV)
        plt.plot(
            np.linspace(0, kwargs["T"] / br.ms, len(V_mV)),
            V_mV,
            c="black",
            linewidth=0.2,
        )
    # plt.vlines(sim['DCN_trs'][0]/br.ms, -75, 0, 'black')
    # for t_ms in sim['sp_DCN'].spike_trains()[idx]/br.ms:
    #     plt.axvspan(t_ms, t_ms+2, color='grey', zorder=10)
    # plt.ylim([-75, 0])
    # plt.xlim([0, sim.t/br.ms])
    plt.xlabel("time (ms)")
    plt.ylabel("$V_m$ (mV)")
    plt.title(f"rate = {np.mean([len(tr) for tr in DCN_trs])/kwargs['T']/br.Hz:.0f} Hz")
    analysis.set_lims(
        [xlower_ms, pause_ms[0], kwargs["T"] / br.ms],
        [-70, 0],
        minmax=(False, True),
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{plotname}_DCN trace.png")

    # DCN raster plot
    plt.figure(figsize=(9 / 3, 4 / 3))
    # PC_trs = np.take_along_axis(np.array([sim['PC_trs'][i]/br.ms for i in range(len(input_sizes_nS))]), np.argsort(input_sizes_nS), axis=0)

    for idx in range(len(DCN_trs)):
        plt.vlines(DCN_trs[idx] / br.ms, idx, idx + 1, "black")
    # plt.xlim([900, 1000])
    plt.xlabel("Time (ms)")
    plt.ylabel("Number of trials")
    analysis.set_lims(
        [xlower_ms, pause_ms[0], kwargs["T"] / br.ms],
        [0, len(sims)],
        minmax=(False, True),
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{plotname}_raster_DCN.png")

    # DCN normalised rate
    plt.figure(figsize=(9 / 3, 4 / 3))
    hist, bins = np.histogram(
        np.concatenate([tr / br.ms for tr in DCN_trs]),
        np.arange(0, kwargs["T"] / br.ms, 10 * br.defaultclock.dt / br.ms),
    )
    bin_centres = (bins[1:] + bins[:-1]) / 2
    plt.plot(bin_centres, hist / (hist[-1] or 1), "k")
    plt.xlabel("Time (ms)")
    plt.ylabel("Firing rate (norm.)")
    analysis.set_lims(
        [xlower_ms, pause_ms[0], kwargs["T"] / br.ms],
        [0, max(hist[bin_centres >= xlower_ms]) / (hist[-1] or 1)],
        divisors=(0, 1),
        minmax=(False, True),
    )
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{plotname}_rate_DCN.png")

    # change in norm firing rate vs pause synchrony

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = eval(" ".join(sys.argv[1:]))
