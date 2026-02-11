
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon

def read_ocean_transects(data_dir=os.getcwd()):

    ds = nc4.Dataset(os.path.join(data_dir,'Atlantic_Transect_Profiles.nc'))

    latitude = ds.variables['latitude'][:]
    Z = ds.variables['Z'][:]
    theta = ds.variables['Theta'][:,:]
    salt = ds.variables['Salt'][:,:]
    depth = ds.variables['depth'][:]

    ds.close()

    return(latitude, Z, depth, theta, salt)

def plot_crosssection(latitude, Z, depth, theta_grid, salt_grid):
    theta_metadata = {'var_name': 'Theta',
                     'long_name': 'Temperature',
                     'vmin': -2,
                     'vmax': 30,
                     'cmap': 'turbo',
                     'units': '$^{\\circ}$C',
                     'contour_ticks': np.arange(5, 30, 5)}

    salt_metadata = {'var_name': 'Salt',
                     'long_name': 'Salinity',
                     'vmin': 32,
                     'vmax': 36.5,
                     'cmap': 'viridis',
                     'units': 'psu',
                     'contour_ticks': np.arange(32,36.5,0.5)}

    fig = plt.figure(figsize=(10, 5))

    gs = GridSpec(1,2, left=0.09, right=0.95, bottom=0.05, top=0.96,hspace=0.03)

    top_bathy = np.column_stack([latitude,depth])
    bottom_bathy = np.column_stack([latitude,6500*np.ones_like(depth)])
    bathy_polygon = np.vstack([top_bathy, np.flipud(bottom_bathy)])
    bathy = Polygon(bathy_polygon, facecolor='silver', edgecolor='k', zorder=10)
    bathy_2 = Polygon(bathy_polygon, facecolor='silver', edgecolor='k', zorder=10)

    ####################################################################################
    # Plotting Theta

    ax3 = fig.add_subplot(gs[0, 0])
    C = plt.pcolormesh(latitude, Z, theta_grid,
                   vmin=theta_metadata['vmin'], vmax=theta_metadata['vmax'], cmap=theta_metadata['cmap'])
    ax3.contourf(latitude, Z, theta_grid, 1000,
                 vmin=theta_metadata['vmin'], vmax=theta_metadata['vmax'], cmap=theta_metadata['cmap'])
    ax3.contour(latitude, Z, theta_grid, levels=theta_metadata['contour_ticks'], colors='k', linewidths=0.75,
                 vmin=theta_metadata['vmin'], vmax=theta_metadata['vmax'])
    ax3.add_patch(bathy_2)
    ax3.set_ylabel('Depth (m)')
    ax3.plot(latitude,depth,'k-',linewidth=0.5)
    ax3.set_ylim([6200,0])
    ax3.set_xticks([-60, -30, 0, 30, 60])
    ax3.set_xlabel('Latitude')
    ax3.set_title('Temperature')

    plt.colorbar(C, ax=ax3, orientation='horizontal', label=theta_metadata['units'])

    ####################################################################################
    # Plotting Salt

    ax4 = fig.add_subplot(gs[0, 1])
    C = plt.pcolormesh(latitude, Z, salt_grid,
                   vmin=salt_metadata['vmin'], vmax=salt_metadata['vmax'], cmap=salt_metadata['cmap'])
    ax4.contourf(latitude, Z, salt_grid, 1000,
                    vmin=salt_metadata['vmin'], vmax=salt_metadata['vmax'], cmap=salt_metadata['cmap'])
    ax4.contour(latitude, Z, salt_grid, levels=salt_metadata['contour_ticks'], colors='k', linewidths=0.75,
                    vmin=salt_metadata['vmin'], vmax=salt_metadata['vmax'])
    ax4.add_patch(bathy)
    ax4.set_ylabel('Depth (m)')
    ax4.plot(latitude,depth,'k-',linewidth=0.5)
    ax4.set_ylim([6200, 0])
    ax4.set_xticks([-60, -30, 0, 30, 60])
    ax4.set_xlabel('Latitude')
    ax4.set_title('Salinity')

    plt.colorbar(C, ax=ax4, orientation='horizontal', label=salt_metadata['units'])

    # plt.savefig(output_file)
    # plt.close(fig)

    plt.show()

def plot_classification_crosssection(latitude, Z, depth, WaterMassIndices, watermasses_long):

    fig = plt.figure(figsize=(8, 5))
    
    gs = GridSpec(1,1, left=0.09, right=0.95, bottom=0.05, top=0.96,hspace=0.03)
    
    top_bathy = np.column_stack([latitude,depth])
    bottom_bathy = np.column_stack([latitude,6500*np.ones_like(depth)])
    bathy_polygon = np.vstack([top_bathy, np.flipud(bottom_bathy)])
    bathy = Polygon(bathy_polygon, facecolor='silver', edgecolor='k', zorder=10)
    bathy_2 = Polygon(bathy_polygon, facecolor='silver', edgecolor='k', zorder=10)
    
    ####################################################################################
    # Plotting results
    
    ax3 = fig.add_subplot(gs[0, 0])
    C = plt.pcolormesh(latitude, Z, WaterMassIndices, cmap='tab10',
                       alpha=0.5, vmin=0, vmax=len(watermasses_long))
    # ax4.contour(latitude, Z, salt_grid, levels=salt_metadata['contour_ticks'], colors='k', linewidths=0.75,
    #                 vmin=salt_metadata['vmin'], vmax=salt_metadata['vmax'])
    ax3.add_patch(bathy)
    ax3.set_ylabel('Depth (m)')
    ax3.plot(latitude,depth,'k-',linewidth=0.5)
    ax3.set_ylim([6200, 0])
    ax3.set_xticks([-60, -30, 0, 30, 60])
    ax3.set_xlabel('Latitude')
    ax3.set_title('Water Mass Classifications')
    
    cbar = plt.colorbar(C, ax=ax3, #orientation='horizontal',
                        ticks=np.arange(len(watermasses_long))+0.5)
    cbar.set_ticklabels(watermasses_long)
    # cbar.ax.tick_params(rotation=90)
    
    plt.show()


def plot_classification_crosssections(latitude, Z, depth, WaterMassIndices_knn, WaterMassIndices_dt, watermasses_long):
    fig = plt.figure(figsize=(10, 5))

    gs = GridSpec(1, 2, left=0.09, right=0.95, bottom=0.05, top=0.96, hspace=0.03)

    top_bathy = np.column_stack([latitude, depth])
    bottom_bathy = np.column_stack([latitude, 6500 * np.ones_like(depth)])
    bathy_polygon = np.vstack([top_bathy, np.flipud(bottom_bathy)])
    bathy = Polygon(bathy_polygon, facecolor='silver', edgecolor='k', zorder=10)
    bathy_2 = Polygon(bathy_polygon, facecolor='silver', edgecolor='k', zorder=10)

    ####################################################################################
    # Plotting KNN results

    ax3 = fig.add_subplot(gs[0, 0])
    C = plt.pcolormesh(latitude, Z, WaterMassIndices_knn, cmap='tab10',
                       alpha=0.5, vmin=0, vmax=len(watermasses_long))
    # ax4.contour(latitude, Z, salt_grid, levels=salt_metadata['contour_ticks'], colors='k', linewidths=0.75,
    #                 vmin=salt_metadata['vmin'], vmax=salt_metadata['vmax'])
    ax3.add_patch(bathy)
    ax3.set_ylabel('Depth (m)')
    ax3.plot(latitude, depth, 'k-', linewidth=0.5)
    ax3.set_ylim([6200, 0])
    ax3.set_xticks([-60, -30, 0, 30, 60])
    ax3.set_xlabel('Latitude')
    ax3.set_title('KNN')

    cbar = plt.colorbar(C, ax=ax3, orientation='horizontal',
                        ticks=np.arange(len(watermasses_long)) + 0.5)
    cbar.set_ticklabels(watermasses_long)
    cbar.ax.tick_params(rotation=90)

    ####################################################################################
    # Plotting DT results

    ax4 = fig.add_subplot(gs[0, 1])
    C = plt.pcolormesh(latitude, Z, WaterMassIndices_dt, cmap='tab10',
                       alpha=0.5, vmin=0, vmax=len(watermasses_long))
    # ax4.contour(latitude, Z, salt_grid, levels=salt_metadata['contour_ticks'], colors='k', linewidths=0.75,
    #                 vmin=salt_metadata['vmin'], vmax=salt_metadata['vmax'])
    ax4.add_patch(bathy_2)
    ax4.set_ylabel('Depth (m)')
    ax4.plot(latitude, depth, 'k-', linewidth=0.5)
    ax4.set_ylim([6200, 0])
    ax4.set_xticks([-60, -30, 0, 30, 60])
    ax4.set_xlabel('Latitude')
    ax4.set_title('Decision Tree')

    cbar = plt.colorbar(C, ax=ax4, orientation='horizontal',
                        ticks=np.arange(len(watermasses_long)) + 0.5)
    cbar.set_ticklabels(watermasses_long)
    cbar.ax.tick_params(rotation=90)

    plt.show()
