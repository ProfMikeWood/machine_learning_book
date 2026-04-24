
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt

def read_turbulence_profiles(file_path):
    transect_sets = {}
    ds = nc4.Dataset(file_path)
    for transect_number in range(1,11):
        transect_sets[transect_number] = {}
        grp = ds.groups['transect_'+str(transect_number)]
        for profile_number in range(1,10):
            profile = grp.groups['profile_'+str(profile_number)]
            pressure = profile.variables['pressure']
            temperature = profile.variables['temperature']
            epsilon = profile.variables['epsilon']
            N2 = profile.variables['N2']
            distance = profile.variables['distance']
            transect_sets[transect_number][profile_number] = np.column_stack([pressure, epsilon, N2, temperature, distance])
    ds.close()
    column_names = ['pressure', 'epsilon', 'N2', 'temperature', 'distance']
    return transect_sets, column_names

def plot_transect_profiles(transect_sets, variable_name):

    if variable_name=='epsilon':
        variable_label = 'Turbulent Dissipation Rate ($\\varepsilon$, W/kg$^{-1}$)'
        cbar_label = 'log$_{10}$($\\varepsilon$)'
        col = 1
        vmin = -9
        vmax = -3
        cmap = 'turbo'
    elif variable_name=='N2':
        variable_label = 'Buoyancy Frequency Squared (N$^2$, s$^{-2}$)'
        cbar_label = 'log$_{10}$(N$^2$)'
        col = 2
        vmin = -6
        vmax = -3
        cmap = 'turbo'
    elif variable_name=='temperature':
        variable_label = 'Temperature (°C)'
        cbar_label = 'Temperature (°C)'
        col = 3
        vmin = 10
        vmax = 16
        cmap = 'turbo'
    else:
        raise ValueError('Invalid variable name. Must be one of: epsilon, N2, temperature.')

    fig = plt.figure(figsize=(8,10))

    gs = fig.add_gridspec(5, 2, wspace=0.1, hspace=0.3,
                          left=0.1, right=0.85, top=0.91, bottom=0.05)

    for transect_number in transect_sets.keys():
        ax = fig.add_subplot(gs[transect_number-1])
        for profile_number in transect_sets[transect_number].keys():
            profile = transect_sets[transect_number][profile_number]
            profile = profile[profile[:,col]!=0, :]
            profile = profile[~np.isnan(profile[:,col]), :]
            if variable_name in ['epsilon', 'N2']:
                plt.scatter(profile[:,4], profile[:,0], c=np.log10(profile[:,col]),
                        cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                plt.scatter(profile[:,4], profile[:,0], c=profile[:,col],
                            cmap=cmap, vmin=vmin, vmax=vmax)
        plt.ylim(40,0)
        plt.title('Transect '+str(transect_number))
        if transect_number in [9, 10]:
            plt.xlabel('Distance Along Transect (km)')
        else:
            ax.set_xticklabels([])
        if transect_number == 5:
            plt.ylabel('Pressure (dbar)')
        if transect_number % 2 == 0:
            ax.set_yticklabels([])

    # add a colorbar axis
    cax = fig.add_axes([0.86, 0.3, 0.02, 0.4])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    plt.show()


def plot_clustering_results(transect_sets, C, TP, n_clusters, plot_cluster='all'):

    clustering_results = {}
    for transect_number in transect_sets.keys():
        clustering_results[transect_number] = {}
        for profile_number in transect_sets[transect_number].keys():
            transect_profile_indices = np.logical_and(TP[:,0] == transect_number, TP[:,1] == profile_number)
            ids = C[transect_profile_indices]
            clustering_results[transect_number][profile_number] = ids

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(8,10))

    gs = fig.add_gridspec(5, 2, wspace=0.1, hspace=0.3,
                          left=0.1, right=0.85, top=0.91, bottom=0.05)

    if plot_cluster=='all':
        cluster_values_unique = np.unique(C)
    else:
        cluster_values_unique = [plot_cluster]

    for transect_number in transect_sets.keys():
        ax = fig.add_subplot(gs[transect_number-1])
        for profile_number in transect_sets[transect_number].keys():
            profile = transect_sets[transect_number][profile_number]

            cluster_values = clustering_results[transect_number][profile_number]
            profile = profile[:np.shape(cluster_values)[0], :]

            plt.plot(profile[:,4], profile[:,0], '.', color='silver', markersize=4, zorder=1)

            for v, value in enumerate(cluster_values_unique):
                profile_subset = profile[cluster_values == value, :]
                if profile_number == 1:
                    plt.scatter(profile_subset[:,4], profile_subset[:,0], color=colors[value % len(colors)],
                                label='Cluster '+str(value), zorder=2)
                else:
                    plt.scatter(profile_subset[:,4], profile_subset[:,0], color=colors[value % len(colors)], zorder=2)
        plt.ylim(40,0)
        plt.title('Transect '+str(transect_number))
        if transect_number in [9, 10]:
            plt.xlabel('Distance Along Transect (km)')
        else:
            ax.set_xticklabels([])
        if transect_number == 5:
            plt.ylabel('Pressure (dbar)')
        if transect_number % 2 == 0:
            ax.set_yticklabels([])

    plt.show()


