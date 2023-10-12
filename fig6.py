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

# PLOTPATH = 'figs/fig6/20230116_'
# PLOTPATH = 'figs/fig6/20230113_'
# PLOTPATH = 'figs/fig6/20230119_'
# PLOTPATH = 'figs/fig6/20230123_'
# PLOTPATH = 'figs/fig6/20230327_'
PLOTPATH = "figs/fig6/20230415_"

G_nS = fig2.G_nS
FINAL_PARAMS = fig2.FINAL_PARAMS  # not used

UNIFORM_PARAMS = {  # These were also used for the different-size inputs
    "mean_PC_ISI": 1 / (80 * br.Hz),
    "C_m": 200 * br.pF,
    "N_extE_DCN": 23650,
    "gL": 5 * br.nS,
    "V_L": -10 * br.mV,
}


# for /l %i in (1 1 10) do for /l %j in (1 1 4) do python fig6.py run_synchrony_systematic(seed=%i, choose=%j, plot=False)


def run_synchrony_systematic(**kwargs):
    kwargs = {
        **{
            "seed": 2,
            "plot": True,
            "runsim": True,
        },
        **kwargs,
    }

    nChoose = kwargs.get("choose", 2)
    fname = f'{PLOTPATH}_fig6_choose_{nChoose}_seed_{kwargs["seed"]}.csv'

    if kwargs["runsim"]:
        totalconductance_nS = 200
        with analysis.seed(kwargs["seed"]):
            input_sizes_nS = []
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
            input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
            input_sizes_nS.sort()
        kwargs_list = [
            {**{"sync_indices": np.array(sync_indices)}, **kwargs}
            for sync_indices in itertools.combinations(
                range(len(input_sizes_nS)), nChoose
            )
        ]
        with Pool(16, initializer=np.random.seed) as p:
            print(kwargs_list)
            sim_list = list(
                tqdm(
                    analysis.starstar(p.imap, run_synchrony, kwargs_list),
                    total=len(kwargs_list),
                    disable=False,
                )
            )
        df_data = {key: [] for key in ["DCN rate (Hz)", "sync size (nS)"]}
        for kwargs, sim in zip(kwargs_list, sim_list):
            df_data["DCN rate (Hz)"].append(sim["DCN_rate_Hz"][0])
            df_data["sync size (nS)"].append(sim["input_sizes_nS"][0])
        df = pd.DataFrame(df_data)
        df.to_csv(fname)
    else:
        df = pd.read_csv(fname, index_col=0)

    if kwargs["plot"]:
        sns.pairplot(df)
        plt.show()


def run_synchrony_experiment(**kwargs):
    """
    with three input sizes

    16 x 3 nS, 12 x 8 nS, 2 x 30 nS

    """
    kwargs = {
        "record": False,
        "T": 10 * br.second,
        "report": None,
        "out": {
            "DCN_rate_Hz": "[len(tr) for tr in net['sp_DCN'].spike_trains().values()]/net.t/br.Hz",
            "gI_mean_nS": "np.mean(gI.values.T*1e9, axis=1)",
            "gI_std_nS": "np.std(gI.values.T*1e9, axis=1)",
        },
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs,
    }

    kwargs_list = []

    size_nS = {"small": 3, "medium": 12, "large": 30}
    N_inputs = {"small": 16, "medium": 8, "large": 2, "all": 2}

    # baseline
    kwargs_list = [
        {
            **kwargs,
            "input_sizes": (
                [0]
                + [size_nS["small"] for _ in range(N_inputs["small"])]
                + [size_nS["medium"] for _ in range(N_inputs["medium"])]
                + [size_nS["large"] for _ in range(N_inputs["large"])]
            )
            * br.nS,
            "N_synced": 0,
            "sync_type": "all",
        }
    ]

    # # sync small inputs
    # for N_synced in range(1,1+N_inputs['small']):
    #     input_sizes_nS = ([size_nS['small']*N_synced]
    #                       + [size_nS['small'] for _ in range(N_inputs['small']-N_synced)]
    #                       + [size_nS['medium'] for _ in range(N_inputs['medium'])]
    #                       + [size_nS['large'] for _ in range(N_inputs['large'])]
    #                       )
    #     kwargs_list.append({
    #         **kwargs,
    #         'input_sizes': input_sizes_nS*br.nS,
    #         'N_synced': N_synced,
    #         'sync_type': 'small',
    #         })

    # # sync medium inputs
    # for N_synced in range(1,1+N_inputs['medium']):
    #     input_sizes_nS = ([size_nS['medium']*N_synced]
    #                       + [size_nS['small'] for _ in range(N_inputs['small'])]
    #                       + [size_nS['medium'] for _ in range(N_inputs['medium']-N_synced)]
    #                       + [size_nS['large'] for _ in range(N_inputs['large'])]
    #                       )
    #     kwargs_list.append({
    #         **kwargs,
    #         'input_sizes': input_sizes_nS*br.nS,
    #         'N_synced': N_synced,
    #         'sync_type': 'medium',
    #         })

    # # sync large inputs
    # for N_synced in range(1,1+N_inputs['large']):
    #     input_sizes_nS = ([size_nS['large']*N_synced]
    #                       + [size_nS['small'] for _ in range(N_inputs['small'])]
    #                       + [size_nS['medium'] for _ in range(N_inputs['medium'])]
    #                       + [size_nS['large'] for _ in range(N_inputs['large']-N_synced)]
    #                       )
    #     kwargs_list.append({
    #         **kwargs,
    #         'input_sizes': input_sizes_nS*br.nS,
    #         'N_synced': N_synced,
    #         'sync_type': 'large',
    #         })

    # sync all inputs (50%)
    input_sizes_nS = (
        [sum([size_nS[x] * N_inputs[x] for x in ["small", "medium", "large"]]) / 2]
        + [size_nS["small"] for _ in range(N_inputs["small"] // 2)]
        + [size_nS["medium"] for _ in range(N_inputs["medium"] // 2)]
        + [size_nS["large"] for _ in range(N_inputs["large"] // 2)]
    )
    kwargs_list.append(
        {
            **kwargs,
            "input_sizes": input_sizes_nS * br.nS,
            "N_synced": 1,
            "sync_type": "all",
        }
    )

    with Pool(16, initializer=np.random.seed) as p:
        sim_list = list(
            tqdm(
                analysis.starstar(p.imap, fig2.run_differentsizeinputs, kwargs_list),
                total=len(kwargs_list),
                disable=False,
            )
        )

    df_data = {
        key: []
        for key in [
            "DCN rate (Hz)",
            "DCN rate (norm.)",
            "sync percent",
            "sync size (nS)",
            "gI CV",
            "gI mean (nS)",
            "gI std (nS)",
        ]
    }

    baseline_rate = sim_list[0]["DCN_rate_Hz"][0]

    for kwargs, sim in zip(kwargs_list, sim_list):
        df_data["DCN rate (Hz)"].append(sim["DCN_rate_Hz"][0])
        df_data["DCN rate (norm.)"].append(sim["DCN_rate_Hz"][0] / baseline_rate)
        df_data["sync percent"].append(
            100 * kwargs["N_synced"] / N_inputs[kwargs["sync_type"]]
        )
        df_data["sync size (nS)"].append(kwargs["input_sizes"][0] / br.nS)
        df_data["gI CV"].append(sim["gI_std_nS"][0] / sim["gI_mean_nS"][0])
        df_data["gI mean (nS)"].append(sim["gI_mean_nS"][0])
        df_data["gI std (nS)"].append(sim["gI_std_nS"][0])
    df = pd.DataFrame(df_data)

    df.to_csv(f"{PLOTPATH}fig6_experiment.csv")

    g = sns.PairGrid(
        df,
        y_vars=["DCN rate (Hz)"],
        x_vars=[
            "sync percent",
            "sync size (nS)",
        ],
        # height=1.5, aspect=.5
    )
    g.map(
        sns.scatterplot,
        # size=1
        alpha=0.5,
    )
    g.add_legend()

    plt.show()


def run_synchrony_random(**kwargs):
    kwargs_list = [
        {
            **{
                "seed": np.random.randint(1, 10),
                # 'N_synced': 2,
            },
            **kwargs,
        }
        for _ in range(kwargs.get("nreps", 16))
    ]
    kwargs_list.extend(
        [
            {
                **{
                    "seed": seed,
                    "N_synced": 1,
                },
                **kwargs,
            }
            for seed in range(1, 11)
        ]
    )
    with Pool(16, initializer=np.random.seed) as p:
        sim_list = list(
            tqdm(
                analysis.starstar(p.imap, run_synchrony, kwargs_list),
                total=len(kwargs_list),
                disable=False,
            )
        )

    df_data = {
        key: []
        for key in [
            "DCN rate (Hz)",
            "DCN rate (norm.)",
            "sync percent",
            "sync size (nS)",
            "seed",
            "gI CV",
            "gI mean (nS)",
            "gI std (nS)",
        ]
    }

    baseline_rates = [sim["DCN_rate_Hz"][0] for sim in sim_list[-10:]]

    for kwargs, sim in zip(kwargs_list, sim_list):
        df_data["DCN rate (Hz)"].append(sim["DCN_rate_Hz"][0])
        df_data["DCN rate (norm.)"].append(
            sim["DCN_rate_Hz"][0] / baseline_rates[kwargs["seed"] - 1]
        )
        df_data["sync percent"].append(sim["percent"])
        df_data["sync size (nS)"].append(sim["input_sizes_nS"][0])
        df_data["seed"].append(kwargs["seed"])
        df_data["gI CV"].append(sim["gI_std_nS"][0] / sim["gI_mean_nS"][0])
        df_data["gI mean (nS)"].append(sim["gI_mean_nS"][0])
        df_data["gI std (nS)"].append(sim["gI_std_nS"][0])
    df = pd.DataFrame(df_data)

    df.to_csv(f"{PLOTPATH}fig6_random.csv")

    g = sns.PairGrid(
        df,
        y_vars=["DCN rate (Hz)"],
        x_vars=[
            "sync percent",
            "sync size (nS)",
        ],
        # height=1.5, aspect=.5
        hue="seed",
    )
    g.map(
        sns.scatterplot,
        # size=1
        alpha=0.5,
    )
    g.add_legend()

    plt.show()


def run_synchrony_manyseeds_worker(**kwargs):
    sims = [run_synchrony(**kwargs) for _ in range(kwargs["nreps"])]
    keys = list(sims[0].keys())
    avg_sim = {k: np.mean([sim[k] for sim in sims], axis=0) for k in keys}
    return avg_sim


def run_synchrony_manyseeds(**kwargs):
    seeds = range(kwargs.get("nseeds", 100))
    reps = range(kwargs.get("nreps", 100))
    kwargs_list = [
        {
            **{
                "seed": seed,
                "sync_indices": indices,
                "T": 10 * br.second,
            },
            **kwargs,
        }
        for seed in seeds
        for indices in [[0, 1], [-1, -2], [0]]
        # for rep in reps
    ]

    with Pool(16, initializer=np.random.seed) as p:
        sim_list = list(
            tqdm(
                analysis.starstar(
                    p.imap,
                    run_synchrony_manyseeds_worker,  # run_synchrony,
                    kwargs_list,
                ),
                total=len(kwargs_list),
                disable=False,
            )
        )

    df_data = {
        key: []
        for key in [
            "DCN rate (Hz)",
            "DCN rate (norm.)",
            "two_sync",
            "sync percent",
            "sync size (nS)",
            "seed",
            "gI CV",
            "gI mean (nS)",
            "gI std (nS)",
        ]
    }

    baseline_rates = [
        np.mean(
            [
                sim["DCN_rate_Hz"][0]
                for kwargs, sim in zip(kwargs_list, sim_list)
                if len(kwargs["sync_indices"]) == 1 and kwargs["seed"] == seed
            ]
        )
        for seed in seeds
    ]

    for kwargs, sim in zip(kwargs_list, sim_list):
        df_data["DCN rate (Hz)"].append(sim["DCN_rate_Hz"][0])
        df_data["DCN rate (norm.)"].append(
            sim["DCN_rate_Hz"][0] / baseline_rates[kwargs["seed"]]
        )
        df_data["two_sync"].append(
            (
                "none"
                if len(kwargs["sync_indices"]) == 1
                else ("smallest" if kwargs["sync_indices"][0] == 0 else "largest")
            )
        )
        df_data["sync percent"].append(sim["percent"])
        df_data["sync size (nS)"].append(sim["input_sizes_nS"][0])
        df_data["seed"].append(kwargs["seed"])
        df_data["gI CV"].append(sim["gI_std_nS"][0] / sim["gI_mean_nS"][0])
        df_data["gI mean (nS)"].append(sim["gI_mean_nS"][0])
        df_data["gI std (nS)"].append(sim["gI_std_nS"][0])
    df = pd.DataFrame(df_data)

    df.to_csv(f"{PLOTPATH}fig6_twolargestsmallest.csv")

    for label in ["DCN rate (Hz)", "DCN rate (norm.)"]:
        df_label = (
            df.groupby(["seed", "two_sync"])
            .mean()
            .reset_index()
            .pivot(index="seed", columns="two_sync", values=label)
        )
        df_label.to_csv(f"{PLOTPATH}fig6_twolargestsmallest_{label}.csv")

    # g = sns.PairGrid(df,
    #                   y_vars=['DCN rate (Hz)'
    #                         ],
    #                   x_vars=['sync percent',
    #                   'sync size (nS)',],
    #                    # height=1.5, aspect=.5
    #                    hue='seed',
    #                    )
    # g.map(sns.scatterplot,
    #       # size=1
    #       alpha=.5,
    #       )
    # g.add_legend()

    # plt.show()


def run_synchrony(**kwargs):
    kwargs = {
        **{
            "record": False,
            "T": 10 * br.second,
            "mean_PC_ISI": 1 / (80 * br.Hz),
            "report": None,
            "out": {
                "DCN_rate_Hz": "[len(tr) for tr in net['sp_DCN'].spike_trains().values()]/net.t/br.Hz",
                "gI_mean_nS": "np.mean(gI.values.T*1e9, axis=1)",
                "gI_std_nS": "np.std(gI.values.T*1e9, axis=1)",
            },
            "seed": 1,
        },
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs,
    }

    input_sizes_nS, sync_indices, N_inputs = sync_input_sizes(**kwargs)
    percent = 100 * len(sync_indices) / N_inputs
    kwargs["input_sizes"] = input_sizes_nS * br.nS

    sim = fig2.run_differentsizeinputs(**kwargs)
    sim["input_sizes_nS"] = input_sizes_nS
    sim["percent"] = percent
    return sim


# for /l %j in (1 1 8) do for /l %i in (1 1 100) do echo %j %i & python fig6.py save_synchrony_xcorr(seed=%j, nreps=160)
def save_synchrony_xcorr(**kwargs):
    kwargs = {
        **{
            "seed": 8,
        },
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs,
    }
    input_sizes_nS, sync_indices, N_inputs = sync_input_sizes(**kwargs)
    kwargs["input_sizes"] = input_sizes_nS * br.nS
    kwargs["fname"] = (
        f'{PLOTPATH}xcorr_seed_{kwargs["seed"]}'
        f'_sync_indices_{"_".join(map(str,sync_indices))}.dat'
    )
    fig2.save_xcorr_parallel(**kwargs)


def plot_synchrony_xcorr(**kwargs):
    """
    save the cross-correlograms of the synchronised input

    notes: input_sizes_nS is before syncing, input_sizes is after
    """
    import seaborn as sns

    # xcorr stats
    avg_window = kwargs.get("moving_average_window", 10)
    # seed = kwargs.get('seed', 8)
    seeds = list(range(1, 11))

    df_xcorr_data = {
        key: []
        for key in [
            "sync percent",
            "input size (nS)",  # sync size
            "time (ms)",
            "spike count",
            "seed",
            "spikes (norm.)",
        ]
    }

    df_data = {
        key: []
        for key in [
            "$c_+$ (norm.)",
            "$c_-$ (norm.)",
            r"$w_{supp}$ (ms)",
            "sync percent",
            "input size (nS)",  # sync size
            "seed",
            "spike mean (norm.)",
            "spike std (norm.)",
        ]
    }

    # get files
    from glob import glob
    import re

    for seed in seeds:
        fnames = glob(f"{PLOTPATH}xcorr_seed_{seed}" f"_sync_indices_*.dat")

        totalconductance_nS = 200

        # get input sizes and p
        with analysis.seed(seed):
            input_sizes_nS = []
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
            input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
            # sizes before syncing
            input_sizes_nS.sort()

        for fname in fnames:
            sync_indices_str = re.findall("sync_indices_(.*).dat", fname)[0]
            sync_indices = [int(index) for index in sync_indices_str.split("_")]

            p = len(sync_indices) / len(input_sizes_nS)

            with open(fname, "rb") as f:
                # sizes after syncing
                hist_list, bins, input_sizes, count = pickle.load(f)
            # print(count)

            size_nS = input_sizes[0] / br.nS  # sync_size_nS

            bin_centres = (bins[:-1] + bins[1:]) / 2

            # for hist, size_nS in zip(hist_list, input_sizes/br.nS):
            hist = hist_list[0]
            baseline = fig2.populate_df(df_data, hist, bin_centres, avg_window)
            df_data["input size (nS)"].append(size_nS)
            df_data["sync percent"].append(100 * p)
            df_data["seed"].append(seed)
            df_data["spike mean (norm.)"].append(np.mean(hist) / baseline)
            df_data["spike std (norm.)"].append(np.std(hist) / baseline)

            # for size_idx, size_nS in enumerate(input_sizes/br.nS):
            for spike_count, t_ms in zip(hist_list[0], bin_centres * 1000):
                df_xcorr_data["input size (nS)"].append(size_nS)
                df_xcorr_data["time (ms)"].append(t_ms)
                df_xcorr_data["spike count"].append(spike_count)
                df_xcorr_data["spikes (norm.)"].append(spike_count / baseline)
                df_xcorr_data["sync percent"].append(100 * p)
                df_xcorr_data["seed"].append(seed)

    df_xcorr = pd.DataFrame(df_xcorr_data)
    df_xcorr.to_csv(f"{PLOTPATH}fig6_xcorr.csv")

    # print([f'{k}: {len(df_data[k])}' for k in df_data])
    # return df_data

    df = pd.DataFrame(df_data)
    df.to_csv(f"{PLOTPATH}fig6_xcorr_stats.csv")

    # # plot the aggregate xcorr stats
    # import seaborn as sns
    # # g = sns.pairplot(df)#, hue='vary size', diag_kind='hist')
    # # g.set(xlim=(0,None), ylim=(0,None))

    g = sns.PairGrid(
        df,
        y_vars=[
            "$c_+$ (norm.)",  #'peak corr (norm.)',
            "$c_-$ (norm.)",  #'min corr (norm.)',
            # r'$w_{0.5}$ (ms)', #'width at 0.5 (ms)',
            # r'$w_0$ (ms)', #'width at 0 (ms)',
            r"$w_{supp}$ (ms)",  # width of suppression (ms)
        ],
        x_vars=[
            "input size (nS)",
            "sync percent",
        ],
        height=1.5,
        aspect=2,
        hue="seed",
    )
    g.map(sns.scatterplot, size=0.25)
    # plt.savefig(f'{PLOTPATH}_fig2efg.png')

    plt.show()


# for /l %j in (1 1 10) do for /l %i in (0 1 1) do echo %j %i & python fig6.py save_synchrony_xcorr_twolargestsmallest(seed=%j, sync_largest=%i)
def save_synchrony_xcorr_twolargestsmallest(**kwargs):
    kwargs = {
        **{
            "seed": 8,
        },
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs,
    }
    kwargs["sync_indices"] = [-1, -2] if kwargs.get("sync_largest", True) else [0, 1]
    input_sizes_nS, sync_indices, N_inputs = sync_input_sizes(**kwargs)
    kwargs["input_sizes"] = input_sizes_nS * br.nS
    kwargs["fname"] = (
        f'{PLOTPATH}xcorr_twolargestsmallest_seed_{kwargs["seed"]}'
        f'_sync_indices_{"_".join(map(str,sync_indices))}.dat'
    )
    fig2.save_xcorr_parallel(**kwargs)


def plot_synchrony_xcorr_twolargestsmallest(**kwargs):
    """
    save the cross-correlograms of the synchronised input

    notes: input_sizes_nS is before syncing, input_sizes is after
    """
    import seaborn as sns

    # xcorr stats
    avg_window = kwargs.get("moving_average_window", 10)
    # seed = kwargs.get('seed', 8)
    seeds = list(range(1, 11))

    df_xcorr_data = {
        key: []
        for key in [
            "sync percent",
            "two_sync",
            "input size (nS)",  # sync size
            "time (ms)",
            "spike count",
            "seed",
            "spikes (norm.)",
        ]
    }

    df_data = {
        key: []
        for key in [
            "$c_+$ (norm.)",
            "$c_-$ (norm.)",
            r"$w_{supp}$ (ms)",
            "sync percent",
            "two_sync",
            "input size (nS)",  # sync size
            "seed",
            "spike mean (norm.)",
            "spike std (norm.)",
        ]
    }

    # get files
    from glob import glob
    import re

    for seed in seeds:
        fnames = glob(
            f"{PLOTPATH}xcorr_twolargestsmallest_seed_{seed}" f"_sync_indices_*.dat"
        )

        totalconductance_nS = 200

        # get input sizes and p
        with analysis.seed(seed):
            input_sizes_nS = []
            while sum(input_sizes_nS) < totalconductance_nS:
                input_sizes_nS.append(np.random.choice(G_nS))
            input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
            # sizes before syncing
            input_sizes_nS.sort()

        for fname in fnames:
            sync_indices_str = re.findall("sync_indices_(.*).dat", fname)[0]
            sync_indices = [int(index) for index in sync_indices_str.split("_")]

            p = len(sync_indices) / len(input_sizes_nS)

            with open(fname, "rb") as f:
                # sizes after syncing
                hist_list, bins, input_sizes, count = pickle.load(f)
            # print(count)

            size_nS = input_sizes[0] / br.nS  # sync_size_nS

            bin_centres = (bins[:-1] + bins[1:]) / 2

            # for hist, size_nS in zip(hist_list, input_sizes/br.nS):
            hist = hist_list[0]
            baseline = fig2.populate_df(df_data, hist, bin_centres, avg_window)
            df_data["input size (nS)"].append(size_nS)
            df_data["sync percent"].append(100 * p)
            df_data["two_sync"].append(
                (
                    "none"
                    if len(sync_indices) == 1
                    else ("smallest" if sync_indices[0] == 0 else "largest")
                )
            )
            df_data["seed"].append(seed)
            df_data["spike mean (norm.)"].append(np.mean(hist) / baseline)
            df_data["spike std (norm.)"].append(np.std(hist) / baseline)

            # for size_idx, size_nS in enumerate(input_sizes/br.nS):
            for spike_count, t_ms in zip(hist_list[0], bin_centres * 1000):
                df_xcorr_data["input size (nS)"].append(size_nS)
                df_xcorr_data["time (ms)"].append(t_ms)
                df_xcorr_data["spike count"].append(spike_count)
                df_xcorr_data["spikes (norm.)"].append(spike_count / baseline)
                df_xcorr_data["sync percent"].append(100 * p)
                df_xcorr_data["two_sync"].append(
                    (
                        "none"
                        if len(sync_indices) == 1
                        else ("smallest" if sync_indices[0] == 0 else "largest")
                    )
                )
                df_xcorr_data["seed"].append(seed)

    df_xcorr = pd.DataFrame(df_xcorr_data)
    df_xcorr.to_csv(f"{PLOTPATH}fig6_xcorr_twolargestsmallest.csv")

    # print([f'{k}: {len(df_data[k])}' for k in df_data])
    # return df_data

    df = pd.DataFrame(df_data)
    df.to_csv(f"{PLOTPATH}fig6_xcorr_twolargestsmallest_stats.csv")

    # # plot the aggregate xcorr stats
    # import seaborn as sns
    # # g = sns.pairplot(df)#, hue='vary size', diag_kind='hist')
    # # g.set(xlim=(0,None), ylim=(0,None))

    g = sns.PairGrid(
        df,
        y_vars=[
            "$c_+$ (norm.)",  #'peak corr (norm.)',
            "$c_-$ (norm.)",  #'min corr (norm.)',
            # r'$w_{0.5}$ (ms)', #'width at 0.5 (ms)',
            # r'$w_0$ (ms)', #'width at 0 (ms)',
            r"$w_{supp}$ (ms)",  # width of suppression (ms)
        ],
        x_vars=[
            "input size (nS)",
            "sync percent",
        ],
        height=1.5,
        aspect=2,
        hue="seed",
    )
    g.map(sns.scatterplot, size=0.25)
    # plt.savefig(f'{PLOTPATH}_fig2efg.png')

    plt.show()


def sync_input_sizes(**kwargs):
    # draw inputs
    totalconductance_nS = 200
    with analysis.seed(kwargs["seed"]):
        input_sizes_nS = []
        while sum(input_sizes_nS) < totalconductance_nS:
            input_sizes_nS.append(np.random.choice(G_nS))
        input_sizes_nS[-1] = totalconductance_nS - sum(input_sizes_nS[:-1])
        input_sizes_nS.sort()

    N_inputs = len(input_sizes_nS)
    sync_indices = kwargs.get(
        "sync_indices",
        np.random.choice(
            range(N_inputs),
            size=kwargs.get("N_synced", np.random.randint(2, N_inputs - 1)),
            replace=False,
        ),
    )
    # shortcut: model the synchronised inputs as a single aggregate input
    input_sizes_nS = np.array(input_sizes_nS)
    input_sizes_nS = np.insert(
        np.delete(input_sizes_nS, sync_indices), 0, sum(input_sizes_nS[sync_indices])
    )
    # kwargs['input_sizes'] = input_sizes_nS*br.nS
    return input_sizes_nS, sync_indices, N_inputs


def run_synchrony_uniform(**kwargs):
    kwargs = {
        "record": False,
        "T": 100 * br.second,
        "report": None,
        "out": {
            "DCN_rate_Hz": "[len(tr) for tr in net['sp_DCN'].spike_trains().values()]/net.t/br.Hz",
            "gI_mean_nS": "np.mean(gI.values.T*1e9, axis=1)",
            "gI_std_nS": "np.std(gI.values.T*1e9, axis=1)",
        },
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs,
    }

    total_conductance_nS = 200

    N_inputs = 40

    kwargs_list = []

    for N_synced in range(N_inputs):
        size_nS = total_conductance_nS / N_inputs
        input_sizes_nS = [size_nS * N_synced] + [
            size_nS for _ in range(N_inputs - N_synced)
        ]
        kwargs_list.append(
            {**kwargs, "input_sizes": input_sizes_nS * br.nS, "N_synced": N_synced}
        )

    with Pool(16, initializer=np.random.seed) as p:
        sim_list = list(
            tqdm(
                analysis.starstar(p.imap, fig2.run_differentsizeinputs, kwargs_list),
                total=len(kwargs_list),
                disable=False,
            )
        )

    df_data = {
        key: []
        for key in [
            "DCN rate (Hz)",
            "DCN rate (norm.)",
            "sync percent",
            "sync size (nS)",
            "gI CV",
            "gI mean (nS)",
            "gI std (nS)",
        ]
    }

    baseline_rate = sim_list[0]["DCN_rate_Hz"][0]

    for kwargs, sim in zip(kwargs_list, sim_list):
        df_data["DCN rate (Hz)"].append(sim["DCN_rate_Hz"][0])
        df_data["DCN rate (norm.)"].append(sim["DCN_rate_Hz"][0] / baseline_rate)
        df_data["sync percent"].append(100 * kwargs["N_synced"] / N_inputs)
        df_data["sync size (nS)"].append(kwargs["input_sizes"][0] / br.nS)
        df_data["gI CV"].append(sim["gI_std_nS"][0] / sim["gI_mean_nS"][0])
        df_data["gI mean (nS)"].append(sim["gI_mean_nS"][0])
        df_data["gI std (nS)"].append(sim["gI_std_nS"][0])
    df = pd.DataFrame(df_data)

    df.to_csv(f"{PLOTPATH}fig6_uniform.csv")

    g = sns.PairGrid(
        df,
        y_vars=["DCN rate (Hz)"],
        x_vars=[
            "sync percent",
            "sync size (nS)",
        ],
        # height=1.5, aspect=.5
    )
    g.map(
        sns.scatterplot,
        # size=1
        alpha=0.5,
    )
    g.add_legend()

    plt.show()


# for /l %i in (1 1 40) do python fig6.py save_synchrony_uniform_xcorr(N_synced=%i, nreps=160)
def save_synchrony_uniform_xcorr(**kwargs):
    kwargs = {
        "N_synced": 1,
        # **FINAL_PARAMS,
        **UNIFORM_PARAMS,
        **kwargs,
    }
    total_conductance_nS = 200
    N_inputs = 40
    N_synced = kwargs["N_synced"]
    size_nS = total_conductance_nS / N_inputs
    input_sizes_nS = [size_nS * N_synced] + [
        size_nS for _ in range(N_inputs - N_synced)
    ]
    kwargs["input_sizes"] = input_sizes_nS * br.nS
    kwargs["fname"] = f"{PLOTPATH}fig6_xcorr_uniform_{N_synced}.dat"
    fig2.save_xcorr_parallel(**kwargs)


def plot_synchrony_uniform_xcorr(**kwargs):
    import seaborn as sns

    # xcorr stats
    avg_window = kwargs.get("moving_average_window", 10)

    df_xcorr_data = {
        key: []
        for key in [
            "sync percent",
            "sync size (nS)",
            "time (ms)",
            "spike count",
            "spikes (norm.)",
        ]
    }

    df_data = {
        key: []
        for key in [
            "$c_+$ (norm.)",
            "$c_-$ (norm.)",
            r"$w_{supp}$ (ms)",
            "sync percent",
            "sync size (nS)",
            "spike mean (norm.)",
            "spike std (norm.)",
        ]
    }

    # get files
    from glob import glob
    import re

    fnames = glob(f"{PLOTPATH}fig6_xcorr_uniform_*.dat")

    totalconductance_nS = 200
    N_inputs = 40

    # plot xcorr fn
    plt.figure(figsize=(6 / 2, 6 / 2))

    norm = mpl.colors.Normalize(vmin=0, vmax=totalconductance_nS)
    cmap = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.turbo)
    cmap.set_array([])
    cb = plt.colorbar(cmap)

    for fname in fnames:
        N_synced = int(re.findall("uniform_(.*).dat", fname)[0])

        p = N_synced / N_inputs

        with open(fname, "rb") as f:
            # sizes after syncing
            hist_list, bins, input_sizes, count = pickle.load(f)
        print(count)

        sync_size_nS = input_sizes[0] / br.nS

        bin_centres = (bins[:-1] + bins[1:]) / 2

        hist = hist_list[0]
        plt.plot(
            bin_centres * 1000, hist / (hist[0] or 1), color=cmap.to_rgba(sync_size_nS)
        )

        baseline = fig2.populate_df(df_data, hist, bin_centres, avg_window)
        df_data["sync percent"].append(100 * p)
        df_data["sync size (nS)"].append(sync_size_nS)
        df_data["spike mean (norm.)"].append(np.mean(hist) / baseline)
        df_data["spike std (norm.)"].append(np.std(hist) / baseline)

        # for size_idx, size_nS in enumerate(input_sizes/br.nS):
        for spike_count, t_ms in zip(hist_list[0], bin_centres * 1000):
            df_xcorr_data["time (ms)"].append(t_ms)
            df_xcorr_data["spike count"].append(spike_count)
            df_xcorr_data["spikes (norm.)"].append(spike_count / baseline)
            df_xcorr_data["sync size (nS)"].append(sync_size_nS)
            df_xcorr_data["sync percent"].append(100 * p)

    plt.ylabel("spikes (norm.)")
    plt.xlabel("t (ms)")
    plt.ylim([0, 2])
    plt.tight_layout()

    df_xcorr = pd.DataFrame(df_xcorr_data).sort_values(["sync size (nS)", "time (ms)"])
    df_xcorr.to_csv(f"{PLOTPATH}fig6_uniform_xcorr.csv")

    df = pd.DataFrame(df_data).sort_values("sync size (nS)")
    df.to_csv(f"{PLOTPATH}fig6_uniform_xcorr_stats.csv")

    # # plot the aggregate xcorr stats
    # import seaborn as sns
    # # g = sns.pairplot(df)#, hue='vary size', diag_kind='hist')
    # # g.set(xlim=(0,None), ylim=(0,None))

    g = sns.PairGrid(
        df,
        y_vars=[
            "$c_+$ (norm.)",  #'peak corr (norm.)',
            "$c_-$ (norm.)",  #'min corr (norm.)',
            # r'$w_{0.5}$ (ms)', #'width at 0.5 (ms)',
            # r'$w_0$ (ms)', #'width at 0 (ms)',
            r"$w_{supp}$ (ms)",  # width of suppression (ms)
        ],
        x_vars=["sync size (nS)"],
        height=1.5,
        aspect=2,
        # hue='seed'
    )
    g.map(sns.scatterplot, size=0.25)
    # plt.savefig(f'{PLOTPATH}_fig2efg.png')

    plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = eval(" ".join(sys.argv[1:]))
