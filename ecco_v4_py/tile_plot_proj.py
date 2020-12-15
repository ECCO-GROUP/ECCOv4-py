"""
ECCO v4 Python: tile_plot_proj

This module includes routines for plotting arrays in different
projections.

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py

"""

from __future__ import division,print_function
import numpy as np
import matplotlib.pylab as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
from cartopy._crs import PROJ4_VERSION
import cartopy.feature as cfeature
from .resample_to_latlon import resample_to_latlon
import xarray as xr

from .plot_utils import assign_colormap

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_proj_to_latlon_grid(lons, lats, data,
                             projection_type = 'robin',
                             plot_type = 'pcolormesh',
                             user_lon_0 = 0,
                             user_lat_0 = None,
                             lat_lim = 50,
                             parallels = None,
                             levels = 20,
                             cmap=None,
                             dx=.25,
                             dy=.25,
                             show_colorbar = False,
                             show_grid_lines = True,
                             show_grid_labels = True,
                             show_coastline = True,
                             show_land = True,
                             grid_linewidth = 1,
                             grid_linestyle = '--',
                             subplot_grid=None,
                             less_output=True,
                             custom_background = False,
                             background_name = [],
                             background_resolution = [],
                             radius_of_influence = 100000,
                             **kwargs):
    """Generate a plot of llc data, resampled to lat/lon grid, on specified
    projection.

    Parameters
    ----------
    lons, lats, data : xarray DataArray    :
        give the longitude, latitude values of the grid, and the 2D field to
        be plotted

    projection_type : string, optional
        denote the type of projection, see Cartopy docs.
        options include
            'robin' - Robinson
            'PlateCarree' - flat 2D projection
            'LambertConformal'
            'Mercator'
            'EqualEarth'
            'AlbersEqualArea'
            'cyl' - Lambert Cylindrical
            'ortho' - Orthographic
            'stereo' - polar stereographic projection, see lat_lim for choosing
            'InterruptedGoodeHomolosine'
                North or South

    user_lon_0 : float, optional, default 0 degrees
        denote central longitude

    user_lat_0 : float, optional,
        denote central latitude (for relevant projections only, see Cartopy)

    lat_lim : int, optional, default 50 degrees
        for stereographic projection, denote the Southern (Northern) bounds for
        North (South) polar projection or cutoff for LambertConformal projection

    parallels : float, optional,
        standard_parallels, one or two latitudes of correct scale
        (for relevant projections only, see Cartopy docs)

    levels : int, optional
        number of contours to plot

    cmap : string or colormap object, optional
        denotes to colormap. Default is 'viridis' for data without sign change,
        and 'RdBu_r' for "diverging" data (i.e. positive and negative)

    dx, dy : float, optional
        latitude, longitude spacing for grid resampling

    show_colorbar : logical, optional, default False
        show a colorbar or not,

    show_grid_lines : logical, optional
        True only possible for Mercator or PlateCarree projections

    grid_linewidth : float, optional, default 1.0
        width of grid lines

    grid_linestyle : string, optional, default = '--'
        pattern of grid lines,

    cmin, cmax : float, optional
        minimum and maximum values for colorbar, default is: min/max of data
        if no sign change, otherwise cmax = max(abs(data)), cmin = -cmax
        i.e. centered about zero

    subplot_grid : dict or list, optional
        specifying placement on subplot as
            dict:
                {'nrows': rows_val, 'ncols': cols_val, 'index': index_val}

            list:
                [nrows_val, ncols_val, index_val]

            equates to

                matplotlib.pyplot.subplot(
                    row=nrows_val, col=ncols_val,index=index_val)

    less_output : string, optional
        debugging flag, don't print if True

    radius_of_influence : float, optional.  Default 100000 m
        the radius of the circle within which the input data is search for
        when mapping to the new grid
    """

    cmap, (cmin,cmax) = assign_colormap(data,cmap)

    for key in kwargs:
        if key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        else:
            print("unrecognized argument ", key)

    #%%
    # To avoid plotting problems around the date line, lon=180E, -180W
    # plot the data field in two parts, A and B.
    # part 'A' spans from starting longitude to 180E
    # part 'B' spans the from 180E to 360E + starting longitude.
    # If the starting  longitudes or 0 or 180 it is a special case.
    if user_lon_0 > -180 and user_lon_0 < 180:
        A_left_limit = user_lon_0
        A_right_limit = 180
        B_left_limit =  -180
        B_right_limit = user_lon_0
        center_lon = A_left_limit + 180

        if not less_output:
            print ('-180 < user_lon_0 < 180')

    elif user_lon_0 == 180 or user_lon_0 == -180:
        A_left_limit = -180
        A_right_limit = 0
        B_left_limit =  0
        B_right_limit = 180
        center_lon = 0

        if not less_output:
            print('user_lon_0 ==-180 or 180')

    else:
        raise ValueError('invalid starting longitude')

    #%%
    # the number of degrees spanned in part A and part B
    num_deg_A =  int((A_right_limit - A_left_limit)/dx)
    num_deg_B =  int((B_right_limit - B_left_limit)/dx)

    # find the longitudal limits of part A and B
    lon_tmp_d = dict()
    if num_deg_A > 0:
        lon_tmp_d['A'] = [A_left_limit, A_right_limit]

    if num_deg_B > 0:
        lon_tmp_d['B'] = [B_left_limit, B_right_limit]

    if projection_type == 'stereo' and user_lat_0 == None:
        if lat_lim > 0:
            user_lat_0 = 90
        else:
            user_lat_0 = -90

    # Make projection axis
    (ax,show_grid_labels) = _create_projection_axis(
            projection_type, user_lon_0, user_lat_0, parallels,
            lat_lim, subplot_grid, less_output)


    #%%
    # loop through different parts of the map to plot (if they exist),
    # do interpolation and plot
    f = plt.gcf()
    if not less_output:
        print('len(lon_tmp_d): ',len(lon_tmp_d))
    for key, lon_tmp in lon_tmp_d.items():

        new_grid_lon, new_grid_lat, data_latlon_projection = \
            resample_to_latlon(lons, lats, data,
                               -90+dy, 90-dy, dy,
                               lon_tmp[0], lon_tmp[1], dx,
                               mapping_method='nearest_neighbor',
                               radius_of_influence = radius_of_influence)

        if isinstance(ax.projection, ccrs.NorthPolarStereo) or \
           isinstance(ax.projection, ccrs.SouthPolarStereo) :
            p, gl, cbar = \
                plot_pstereo(new_grid_lon,
                             new_grid_lat,
                             data_latlon_projection,
                             4326, lat_lim,
                             cmin, cmax, ax,
                             plot_type = plot_type,
                             show_colorbar=False,
                             show_coastline = show_coastline,
                             show_land = show_land,
                             circle_boundary=True,
                             cmap=cmap,
                             show_grid_lines=show_grid_labels,
                             custom_background = custom_background,
                             background_name = background_name,
                             background_resolution = background_resolution,
                             less_output=less_output)

        else: # not polar stereo
            p, gl, cbar = \
                plot_global(new_grid_lon,
                            new_grid_lat,
                            data_latlon_projection,
                            4326,
                            cmin, cmax, ax,
                            plot_type = plot_type,
                            show_colorbar = False,
                            cmap=cmap,
                            show_grid_lines = False,
                            show_coastline = show_coastline,
                            show_land = show_land,
                            custom_background = custom_background,
                            background_name = background_name,
                            background_resolution = background_resolution,
                            show_grid_labels = show_grid_labels)

        if show_grid_lines :
            ax.gridlines(crs=ccrs.PlateCarree(),
                                  linewidth=grid_linewidth,
                                  color='black',
                                  alpha=0.5,
                                  linestyle=grid_linestyle,
                                  draw_labels = show_grid_labels,zorder=102)

        if show_land:
            ax.add_feature(cfeature.LAND, zorder=100)

        if show_coastline:
            ax.add_feature(cfeature.COASTLINE,linewidth=0.5,zorder=101)

    ax= plt.gca()

    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
        label=''
        if type(data) == xr.DataArray:
            if 'long_name' in data.attrs:
                label = data.long_name
            elif data.name is not None:
                label = data.name
            if 'units' in data.attrs:
                label+= ' ['+data.units+']'

        cbar.set_label(label)

    #%%
    return f, ax, p, cbar, new_grid_lon, new_grid_lat, data_latlon_projection
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_pstereo(xx,yy, data,
                 data_projection_code, \
                 lat_lim,
                 cmin, cmax, ax,
                 plot_type = 'pcolormesh',
                 show_colorbar=False,
                 circle_boundary = False,
                 grid_linewidth = 1,
                 grid_linestyle = '--',
                 cmap=None,
                 show_grid_lines = False,
                 show_coastline = True,
                 show_land = True,
                 custom_background = False,
                 background_name = [],
                 background_resolution = [],
                 levels = 20,
                 less_output=True):

    # assign cmap default
    if cmap is None:
        if cmin*cmax<0:
            cmap = 'RdBu_r'
        else:
            cmap = 'viridis'


    if isinstance(ax.projection, ccrs.NorthPolarStereo):
        ax.set_extent([-179.5, 180, lat_lim, 90], ccrs.PlateCarree())
        if not less_output:
            print('North Polar Projection')
    elif isinstance(ax.projection, ccrs.SouthPolarStereo):
        ax.set_extent([-179.5, 180, -90, lat_lim], ccrs.PlateCarree())
        if not less_output:
            print('South Polar Projection')
    else:
        raise ValueError('ax must be either ccrs.NorthPolarStereo or ccrs.SouthPolarStereo')

    if not less_output:
        print('lat_lim: ',lat_lim)

    if circle_boundary:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if show_grid_lines :
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          linewidth=grid_linewidth, color='black',
                          alpha=0.5, linestyle=grid_linestyle, zorder=102)
    else:
        gl = []

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        # reproject the data if necessary
        data_crs=ccrs.epsg(data_projection_code)

    if custom_background:
        ax.background_img(name=background_name, resolution=background_resolution)

    p=[]
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs, \
                          vmin=cmin, vmax=cmax, cmap=cmap)

    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,  \
                 vmin=cmin, vmax=cmax, cmap=cmap)

    elif plot_type == 'points':
        p = ax.plot(xx, yy,  'k.', transform=data_crs)

    else:
        raise ValueError('plot_type  must be either "pcolormesh", "contourf", or "points"')

    if show_land and not custom_background:
        ax.add_feature(cfeature.LAND, zorder=100)

    if show_coastline:
        ax.coastlines('110m', linewidth=0.8, zorder=101)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)

    return p, gl, cbar

#%%

def plot_global(xx,yy, data,
                data_projection_code,
                cmin, cmax, ax,
                plot_type = 'pcolormesh',
                show_colorbar=False,
                cmap=None,
                show_grid_lines = True,
                show_grid_labels = True,
                show_coastline = True,
                show_land = True,
                grid_linewidth = 1,
                custom_background = False,
                background_name = [],
                background_resolution = [],
                levels=20):

    # assign cmap default
    if cmap is None:
        if cmin*cmax<0:
            cmap = 'RdBu_r'
        else:
            cmap = 'viridis'

    if show_grid_lines :
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          linewidth=1, color='black',
                          draw_labels = show_grid_labels,
                          alpha=0.5, linestyle='--', zorder=102)
    else:
        gl = []

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs =ccrs.epsg(data_projection_code)

    if custom_background:
        ax.background_img(name=background_name, resolution=background_resolution)

    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs,
                          vmin=cmin, vmax=cmax, cmap=cmap)
    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,
                        vmin=cmin, vmax=cmax, cmap=cmap)
    else:
        raise ValueError('plot_type  must be either "pcolormesh" or "contourf"')


    if show_land and not custom_background:
        ax.add_feature(cfeature.LAND, zorder=100)

    if show_coastline:
        ax.coastlines('110m', linewidth=grid_linewidth, zorder=101)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)

    return p, gl, cbar

# -----------------------------------------------------------------------------

def _create_projection_axis(projection_type,
                            user_lon_0,
                            user_lat_0,
                            parallels,
                            lat_lim,
                            subplot_grid,
                            less_output):

    """Set appropriate axis for projection type
    See plot_proj_to_latlon_grid for input parameter definitions.

    Returns
    -------
    ax :  matplotlib axis object
        defined with the correct projection
    show_grid_labels : logical
        True = show the grid labels, only currently
        supported for PlateCarree and Mercator projections
    """

    if not less_output:
        print('_create_projection_axis: projection_type', projection_type)
        print('_create_projection_axis: user_lon_0, user_lat_0', user_lon_0, user_lat_0)
        print('_create_projection_axis: parallels', parallels)
        print('_create_projection_axis: lat_lim', lat_lim)

    # initialize (optional) subplot variables
    row = []
    col = []
    ind = []

    if subplot_grid is not None:

        if type(subplot_grid) is dict:
            row = subplot_grid['nrows']
            col = subplot_grid['ncols']
            ind = subplot_grid['index']

        elif type(subplot_grid) is list:
            row = subplot_grid[0]
            col = subplot_grid[1]
            ind = subplot_grid[2]

        else:
            raise TypeError('Unexpected subplot_grid type: ',type(subplot_grid))
    else:
        row = 1
        col = 1
        ind = 1

    # Build dictionary of projection_types mapped to Cartopy calls
    proj_dict = {'Mercator':ccrs.Mercator,
             'LambertConformal':ccrs.LambertConformal,
             'AlbersEqualArea':ccrs.AlbersEqualArea,
             'PlateCarree':ccrs.PlateCarree,
             'cyl':ccrs.LambertCylindrical,
             'robin':ccrs.Robinson,
             'ortho': ccrs.Orthographic,
             'InterruptedGoodeHomolosine':ccrs.InterruptedGoodeHomolosine
             }

    # This projection requires proj4 v.>= 5.2.0
    if PROJ4_VERSION>=(5,2,0):
        proj_dict['EqualEarth']=ccrs.EqualEarth

    # stereo special cases
    if projection_type == 'stereo':
        if lat_lim>0:
            proj_dict['stereo']=ccrs.NorthPolarStereo
        else :
            proj_dict['stereo']=ccrs.SouthPolarStereo

    if projection_type not in proj_dict:
        raise NotImplementedError('projection type must be in ',proj_dict.keys())

    # Build dictionary for projection arguments
    proj_args={}
    if user_lon_0 is not None:
        proj_args['central_longitude']=user_lon_0
    if user_lat_0 is not None and projection_type != 'stereo':
        proj_args['central_latitude']=user_lat_0
    if (projection_type == 'LambertConformal') & (lat_lim is not None) :
        proj_args['cutoff']=lat_lim
    if parallels is not None :
        proj_args['standard_parallels']=parallels

    ax = plt.subplot(row, col, ind,
                    projection=proj_dict[projection_type](**proj_args))

    if (projection_type == 'Mercator') | (projection_type== 'PlateCarree'):
        show_grid_labels = True
    else:
        show_grid_labels = False

    if not less_output:
        print('Projection type: ', projection_type)

    return (ax,show_grid_labels)
