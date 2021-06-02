import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import copy


def plot_classifier_time_slice(
    time_slice,
    classifier,
    results,
    data,
    posterior_type="acausal_posterior",
    figsize=(30, 15),
    legend=True,
):

    t = data["position_info"].index / np.timedelta64(1, "s")
    cmap = copy.copy(plt.get_cmap('bone_r'))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.5, 3, 1, 1, 1, 1, 0.25]},
    )
    
    # ax 0
    ripple_lfps = (
        data['ripple_lfps']
        .reset_index(drop=True)
        .set_index(data['ripple_lfps'].index / np.timedelta64(1, "s"))
        .loc[slice(time_slice.values[0], time_slice.values[-1])])
    max_tetrode_ind = np.abs(data['ripple_filtered_lfps']).max().idxmax()
    axes[0].plot(ripple_lfps.index, ripple_lfps.iloc[:, max_tetrode_ind], color='black')
    axes[0].axis("off")
    
    
    # ax 1
    (results[posterior_type]
     .sum('state')
     .where(classifier.is_track_interior_)
     .sel(time=time_slice)
     .plot(
        x="time", y="position", robust=True, ax=axes[1],
        cmap=cmap, vmin=0.0,
    ))

    axes[1].set_ylabel("Position [cm]")

    axes[1].scatter(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .linear_position,
        color="magenta",
        alpha=0.8,
        s=10,
        clip_on=False,
    )
    axes[1].set_xlabel("")

    # ax 2
    probability = results[posterior_type].sum('position')
    h = probability.sel(time=time_slice).plot(
        x="time", hue="state", ax=axes[2], add_legend=False, clip_on=False,
    )
    if legend:
        axes[2].legend(handles=h, labels=probability.state.values.tolist(),
                       bbox_to_anchor=(1.10, 0.8), loc='upper right', ncol=1, fontsize=12)
    axes[2].set_ylabel("Probability")
    axes[2].set_xlabel("")
    
    # ax 3
    multiunit_spikes = pd.DataFrame(
        data['multiunit_spikes'],
        index=data['position_info'].index / np.timedelta64(1, "s")
    ).loc[time_slice]
    
    spike_time_ind, tetrode_ind = np.nonzero(np.asarray(multiunit_spikes))
    axes[3].scatter(multiunit_spikes.index[spike_time_ind], tetrode_ind, clip_on=False, s=1, color='black')
    axes[3].set_ylabel("Tetrodes")

    # ax 4
    multiunit_firing = (
        data["multiunit_firing_rate"]
        .reset_index(drop=True)
        .set_index(
            data["multiunit_firing_rate"].index / np.timedelta64(1, "s"))
    )

    axes[4].fill_between(
        multiunit_firing.loc[time_slice].index.values,
        multiunit_firing.loc[time_slice].values.squeeze(),
        color="black",
    )
    axes[4].set_ylabel("Firing Rate\n[spikes / s]")

    # ax 5
    axes[5].fill_between(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        np.abs(data["position_info"]
               .reset_index()
               .set_index(t)
               .loc[time_slice]
               .tailBase_vel.values.squeeze()),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[5].set_ylabel("Speed \n[cm / s]")
    
    # ax 6
    axes[6].eventplot(
        data['dio'].index[data['dio'].loc[:, ['Din1', 'Din2', 'Din3']].sum(axis=1) > 0]
        / np.timedelta64(1, 's'), color='black')
    axes[6].eventplot(
        data['dio'].index[data['dio'].loc[:, ['Dout4', 'Dout5', 'Dout6']].sum(axis=1) > 0]
        / np.timedelta64(1, 's'), color='red')
    axes[6].set_yticklabels([])
    axes[6].set_ylabel("Beam\nBreak")

    axes[-1].set_xlabel("Time [s]")
    sns.despine()

def plot_classifier_inbound_outbound_time_slice(
    time_slice,
    classifier,
    results,
    data,
    posterior_type="acausal_posterior",
    figsize=(30, 15),
    legend=True,
):

    t = data["position_info"].index / np.timedelta64(1, "s")
    cmap = copy.copy(plt.get_cmap('bone_r'))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )

    # ax 0
    (results[posterior_type]
     .sum('state')
     .where(classifier.is_track_interior_)
     .sel(time=time_slice)
     .plot(
         x="time",
         y="position",
         robust=True,
         ax=axes[0],
         cmap=cmap,
         vmin=0.0,
         add_colorbar=legend,
    ))

    axes[0].set_ylabel("Position [cm]")

    axes[0].scatter(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .linear_position,
        color="magenta",
        alpha=0.8,
        s=10,
        zorder=100,
    )
    axes[0].set_xlabel("")

    # ax 1
    fragmented = (results[posterior_type]
                  .sel(state=['Inbound-Fragmented', 'Outbound-Fragmented'])
                  .sum(['state', 'position'])
                  .assign_coords({'state': 'Fragmented'}))
    probability = (results[posterior_type]
                   .sum('position')
                   .drop_sel(state=['Inbound-Fragmented', 'Outbound-Fragmented']))
    probability = xr.concat((probability, fragmented), dim='state')
    h = probability.sel(time=time_slice).plot(
        x="time", hue="state", ax=axes[1], add_legend=False, clip_on=False,
    )
    if legend:
        axes[1].legend(handles=h, labels=probability.state.values.tolist(),
                       bbox_to_anchor=(1.10, 0.8), loc='upper right', ncol=1, fontsize=12)
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("")
    axes[1].set_ylim((0, 1))

    # ax 2
    multiunit_firing = (
        data["multiunit_firing_rate"]
        .reset_index(drop=True)
        .set_index(
            data["multiunit_firing_rate"].index / np.timedelta64(1, "s"))
    )

    axes[2].fill_between(
        multiunit_firing.loc[time_slice].index.values,
        multiunit_firing.loc[time_slice].values.squeeze(),
        color="black",
    )
    axes[2].set_ylabel("Firing Rate\n[spikes / s]")

    # ax 3
    axes[3].fill_between(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        np.abs(data["position_info"]
               .reset_index()
               .set_index(t)
               .loc[time_slice]
               .tailBase_vel.values.squeeze()),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[3].set_ylabel("Speed \n[cm / s]")
    axes[3].set_xlabel("Time [s]")
    axes[3].ticklabel_format(style='plain', axis='x')
    sns.despine()
    
def plot_local_non_local_time_slice(
    time_slice,
    detector,
    results,
    data,
    posterior_type="acausal_posterior",
    figsize=(30, 15),
):
    t = data["position_info"].index / np.timedelta64(1, "s")
    mask = np.ones_like(detector.is_track_interior_.squeeze(), dtype=np.float)
    mask[~detector.is_track_interior_] = np.nan
    cmap = copy.copy(plt.get_cmap('bone_r'))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )

    # ax 0
    (results[posterior_type].sel(time=time_slice).sum("state") * mask).plot(
        x="time", y="position", robust=True, ax=axes[0], cmap=cmap, vmin=0.0,
    )
    axes[0].set_ylabel("Position [cm]")

    axes[0].set_title("Non-Local Posterior")

    axes[0].plot(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .linear_position,
        color="white",
        linestyle="--",
        linewidth=5,
        alpha=0.8,
    )
    axes[0].set_xlabel("")

    # ax 1
    results[posterior_type].sum("position").sel(
        state="Non-Local", time=time_slice
    ).plot(x="time", ax=axes[1])
    axes[1].set_title("Non-Local Probability")
    axes[1].set_ylabel("Probability")
    axes[1].set_xlabel("")

    # ax 2
    multiunit_firing = (
        data["multiunit_firing_rate"]
        .reset_index(drop=True)
        .set_index(
            data["multiunit_firing_rate"].index / np.timedelta64(1, "s"))
    )

    axes[2].fill_between(
        multiunit_firing.loc[time_slice].index.values,
        multiunit_firing.loc[time_slice].values.squeeze(),
        color="black",
    )
    axes[2].set_ylabel("Firing Rate\n[spikes / s]")
    axes[2].set_title("Multiunit")

    # ax 3
    axes[3].fill_between(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        np.abs(data["position_info"]
               .reset_index()
               .set_index(t)
               .loc[time_slice]
               .tailBase_vel.values.squeeze()),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[3].set_ylabel("tailBase_vel [cm / s]")
    axes[3].set_xlabel("Time [s]")
    sns.despine()

    
def get_neuron_order(classifier):
    n_neurons = len(classifier.place_fields_.neuron)
    idx = classifier.place_fields_.argmax(dim=['encoding_group', 'position'])
    idx = np.stack((idx['encoding_group'].values, idx['position'].values, np.arange(n_neurons)), axis=1)
    idx = idx[np.argsort(idx[:, 1])]
    idx = np.concatenate((idx[idx[:, 0] == 0],
                          idx[idx[:, 0] == 1]))
    
    return pd.DataFrame(idx, columns=['encoding_group', 'position_bin', 'neuron_ind'])


def plot_classifier_spikes(
    time_slice,
    classifier,
    results,
    data,
    posterior_type="acausal_posterior",
    figsize=(30, 15),
    legend=True,
):

    t = data["position_info"].index / np.timedelta64(1, "s")
    cmap = copy.copy(plt.get_cmap('bone_r'))
    cmap.set_bad(color="lightgrey", alpha=1.0)

    fig, axes = plt.subplots(
        7,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.5, 3, 1, 1, 1, 1, 0.25]},
    )
    
    # ax 0
    ripple_lfps = (
        data['ripple_lfps']
        .reset_index(drop=True)
        .set_index(data['ripple_lfps'].index / np.timedelta64(1, "s"))
        .loc[slice(time_slice.values[0], time_slice.values[-1])])
    max_tetrode_ind = np.abs(data['ripple_filtered_lfps']).max().idxmax()
    axes[0].plot(ripple_lfps.index, ripple_lfps.iloc[:, max_tetrode_ind], color='black')
    axes[0].axis("off")

    # ax 1
    (results[posterior_type]
     .sum('state')
     .where(classifier.is_track_interior_)
     .sel(time=time_slice)
     .plot(
         x="time",
         y="position",
         robust=True,
         ax=axes[1],
         cmap=cmap,
         vmin=0.0,
         add_colorbar=legend,
    ))

    axes[1].set_ylabel("Position [cm]")

    axes[1].scatter(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        data["position_info"]
        .reset_index()
        .set_index(t)
        .loc[time_slice]
        .linear_position,
        color="magenta",
        alpha=0.8,
        s=10,
        zorder=100,
        clip_on=False,
    )
    axes[1].set_xlabel("")

    # ax 2
    try:
        fragmented = (results[posterior_type]
                      .sel(state=['Inbound-Fragmented', 'Outbound-Fragmented'])
                      .sum(['state', 'position'])
                      .assign_coords({'state': 'Fragmented'}))
        probability = (results[posterior_type]
                       .sum('position')
                       .drop_sel(state=['Inbound-Fragmented', 'Outbound-Fragmented']))
        probability = xr.concat((probability, fragmented), dim='state')
    except KeyError:
        probability = results[posterior_type].sum('position')
    h = probability.sel(time=time_slice).plot(
        x="time", hue="state", ax=axes[2], add_legend=False, clip_on=False,
    )
    if legend:
        axes[2].legend(handles=h, labels=probability.state.values.tolist(),
                       bbox_to_anchor=(1.10, 0.8), loc='upper right', ncol=1, fontsize=12)
    axes[2].set_ylabel("Probability")
    axes[2].set_xlabel("")
    axes[2].set_ylim((0, 1))
    
    # ax 3
    neuron_order = get_neuron_order(classifier)

    spikes = (
        data["spikes"]
        .reset_index(drop=True)
        .set_index(
            data["spikes"].index / np.timedelta64(1, "s"))
    )
    spikes = spikes.iloc[:, neuron_order.neuron_ind]

    spike_time_ind, neuron_ind = np.nonzero(np.asarray(spikes.loc[time_slice]))
    spike_times = np.asarray(spikes.loc[time_slice].index)[spike_time_ind]
    encoding_group = neuron_order.encoding_group.values[neuron_ind]
    
    for group in np.unique(encoding_group):
        axes[3].scatter(spike_times[encoding_group == group],
                        neuron_ind[encoding_group == group] + 1,
                        zorder=1,
                        marker='|',
                        s=20,
                        linewidth=2,
                        clip_on=False)
    n_neurons = spikes.shape[1]
    axes[3].set_yticks((0, n_neurons))
    axes[3].set_ylim((0, n_neurons))
    axes[3].set_ylabel('Cell ID')
    sns.despine(ax=axes[3], bottom=True)

    # ax 4
    multiunit_firing = (
        data["multiunit_firing_rate"]
        .reset_index(drop=True)
        .set_index(
            data["multiunit_firing_rate"].index / np.timedelta64(1, "s"))
    )

    axes[4].fill_between(
        multiunit_firing.loc[time_slice].index.values,
        multiunit_firing.loc[time_slice].values.squeeze(),
        color="black",
    )
    axes[4].set_ylabel("Firing Rate\n[spikes / s]")

    # ax 5
    axes[5].fill_between(
        data["position_info"].reset_index().set_index(t).loc[time_slice].index,
        np.abs(data["position_info"]
               .reset_index()
               .set_index(t)
               .loc[time_slice]
               .tailBase_vel.values.squeeze()),
        color="lightgrey",
        linewidth=1,
        alpha=0.5,
    )
    axes[5].set_ylabel("Speed \n[cm / s]")
    
    # ax 6
    axes[6].eventplot(
        data['dio'].index[data['dio'].loc[:, ['Din1', 'Din2', 'Din3']].sum(axis=1) > 0]
        / np.timedelta64(1, 's'), color='black')
    axes[6].eventplot(
        data['dio'].index[data['dio'].loc[:, ['Dout4', 'Dout5', 'Dout6']].sum(axis=1) > 0]
        / np.timedelta64(1, 's'), color='red')
    axes[6].set_yticklabels([])
    axes[6].set_ylabel("Beam\nBreak")
    
    axes[-1].set_xlabel("Time [s]")
    axes[-1].ticklabel_format(style='plain', axis='x')
    
    sns.despine()
    
    
def plot_place_fields(classifier, sampling_frequency=500):
    n_neurons = len(classifier.place_fields_.neuron)

    idx = classifier.place_fields_.argmax(dim=['encoding_group', 'position'])
    idx = np.stack((idx['encoding_group'].values, idx['position'].values, np.arange(n_neurons)), axis=1)
    idx = idx[np.argsort(idx[:, 1])]
    idx = np.concatenate((idx[idx[:, 0] == 0],
                          idx[idx[:, 0] == 1]))
    sorted_neuron_idx = idx[:, 2]

    fig, axes = plt.subplots(n_neurons, 1, sharex=True, figsize=(14, n_neurons / 8))

    for neuron_ind, ax in zip(sorted_neuron_idx, axes.flat):
        place_field = (classifier
         .place_fields_
         .isel(neuron=neuron_ind)
         .where(classifier.is_track_interior_) * sampling_frequency
        )
        for encoding_group in place_field.encoding_group:
            ax.fill_between(place_field.position, place_field.sel(encoding_group=encoding_group), alpha=0.8)

        ax.set_xlim((0, classifier.place_fields_.position[-1]))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_yticklabels([])

    sns.despine()
    axes[-1].set_xlabel('Position [cm]')

    for ax in axes[:-1].flat:
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    axes[-1].get_yaxis().set_visible(False)
    axes[-1].spines["left"].set_visible(False)