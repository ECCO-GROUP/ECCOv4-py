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
import cartopy as cartopy
import cartopy.feature as cfeature
from .resample_to_latlon import resample_to_latlon

from .plot_utils import assign_colormap

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def plot_proj_to_latlon_grid(lons, lats, data,
                             projection_type = 'robin',
                             dx=.25, dy=.25,
                             radius_of_influence = 112000,
                             plot_type = 'pcolormesh',
                             cmap = None,
                             cmin = None, 
                             cmax = None,
                             user_lon_0 = 0,
                             user_lat_0 = None,
                             lat_lim = 50,
                             parallels = None,
                             show_coastline = True,
                             show_colorbar = False,
                             show_land = True,
                             show_grid_lines = True,
                             show_grid_labels = False,
                             show_coastline_over_data = True,
                             show_land_over_data = True,
                             grid_linewidth = 1,
                             grid_linestyle = '--',
                             colorbar_label = None,
                             subplot_grid=None,
                             less_output=True,
                             **kwargs):
    
    """Plot a field of data from an arbitrary grid with lat/lon coordinates
    on a geographic projection after resampling it to a regular lat/lon grid.
    

    Parameters
    ----------
    lons, lats : numpy ndarray or xarray DataArrays, required
        the longitudes and latitudes of the data to plot
        
    data : numpy ndarray or xarray DataArray, required
        the field to be plotted
        
    dx, dy : float, optional, default 0.25 degrees
        latitude, longitude spacing of the new lat/lon grid onto which the 
        field 'data' will be resampled.

    radius_of_influence : float, optional, default 112000 m
        to map values from 'data' to the new lat/lon grid, we use use a
        nearest neighbor approach with the constraint that we only use values
        from 'data' that fall within a circle with radius='radius_of_influence'
        from the center of each new lat/lon grid cell. 
        for the llc90, with 1 degree resolution, 
            radius_of_influence = 1/2 x sqrt(2) x 112e3 km 
        would suffice.
        
    projection_type : string, optional
        denote the type of projection, see Cartopy docs.
        options include
            'robin' - Robinson
            'PlateCarree' - flat 2D projection
            'LambertConformal'
            'Mercator'
            'EqualEarth'
            'Mollweide'
            'AlbersEqualArea'
            'cyl' - Lambert Cylindrical
            'ortho' - Orthographic
            'stereo' - polar stereographic projection, see lat_lim for choosing
            'InterruptedGoodeHomolosine'
                North or South
        
    plot_type : string, optional
        denotes type of plot ot make with the data
        options include
            'pcolormesh' - pcolormesh
            'contourf' - filled contour
            'points' - plot points at lat/lon locations
    
    cmap : matplotlib.colors.Colormap, optional, default None
        a colormap for the figure.  
    
    cmin/cmax : floats, optional, default None
        the minimum and maximum values to use for the colormap
        if not specified, use the full range of the data
            
    user_lon_0 : float, optional, default 0 degrees
        denote central longitude

    user_lat_0 : float, optional, default None
        denote central latitude (for relevant projections only, see Cartopy)

    lat_lim : int, optional, default 50 degrees
        for stereographic projection, denote the Southern (Northern) bounds for
        North (South) polar projection or cutoff for LambertConformal projection

    parallels : float, optional,
        standard_parallels, one or two latitudes of correct scale
        (for relevant projections only, see Cartopy docs)

    show_coastline : logical, optional, default True
        show coastline or not
                             
    show_colorbar : logical, optional, default False
        show a colorbar or not,

    show_land : logical, optional, default True
        show land or not

    show_grid_lines : logical, optional, default True
        True only possible for some cartopy projections

    show_grid_labels: logical, optional, default False
        True only possible for some cartopy projections

    show_coastline_over_data : logical, optional, default True
        draw coastline over the data or under the data

    show_land_over_data: logical, optional, default True
        draw land over the data or under the data

    grid_linewidth : float, optional, default 1.0
        width of grid lines

    grid_linestyle : string, optional, default = '--'
        pattern of grid lines,
                
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

    """

    if cmap == None:
        cmap, (new_cmin,new_cmax) = assign_colormap(data, cmap)
    
    if cmin == None:
        cmin = np.nanmin(data[:])
    if cmax == None:
        cmax = np.nanmax(data[:])

    if projection_type == 'stereo' and user_lat_0 == None:
        if lat_lim > 0:
            user_lat_0 = 90
        else:
            user_lat_0 = -90

    # Make projection axis
    ax  = _create_projection_axis(
            projection_type, user_lon_0, user_lat_0, parallels,
            lat_lim, subplot_grid, less_output)
    
    ax.set_global()

    # lat-lon data is EPSG:4326
    # https://spatialreference.org/ref/epsg/wgs-84/
    # +proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs 
    data_epsg_code = 4326 
    
    # get current figure
    f = plt.gcf()
    
    # initialize some variables that may not get defined if there is an error
    p = []
    cbar = []
    gl = []
    
    # Polar Stereographic
    if isinstance(ax.projection, ccrs.NorthPolarStereo) or \
       isinstance(ax.projection, ccrs.SouthPolarStereo) :
          
        new_grid_lon_centers, new_grid_lat_centers,\
        new_grid_lon_edges, new_grid_lat_edges,\
        data_latlon_projection = \
            resample_to_latlon(lons, lats, data,
                               -90, 90, dy,
                               -180, 180, dx,
                               mapping_method='nearest_neighbor',
                               radius_of_influence = radius_of_influence)
            
        if plot_type == 'pcolormesh':
            plot_lons = new_grid_lon_edges
            plot_lats = new_grid_lat_edges
        else:
            plot_lons = new_grid_lon_centers
            plot_lats = new_grid_lat_centers
            
        p, gl, cbar = \
            plot_pstereo(plot_lons, 
                         plot_lats, 
                         data_latlon_projection,
                         data_epsg_code,
                         lat_lim,
                         cmin, cmax, ax,
                         plot_type = plot_type,
                         cmap = cmap,
                         show_coastline = show_coastline,
                         show_colorbar = show_colorbar,
                         show_land = show_land,
                         show_grid_lines = show_grid_lines,
                         show_grid_labels = show_grid_labels,
                         show_coastline_over_data = show_coastline_over_data,
                         show_land_over_data = show_land_over_data,
                         grid_linewidth = grid_linewidth,
                         grid_linestyle = grid_linestyle,
                         colorbar_label = colorbar_label,
                         less_output=less_output)
    
           
    else: # not polar stereographic projection
    
        # To avoid plotting problems around the date line, lon=180E, -180W
        # we may have to break the plotting into two parts
        # also, for some reason cartopy or matplotlib 
        # doesn't like it when the edge of the 
        # longitude array is the same as the edge of the map when 
        # user_lon_0 (center_longitude) is not 0 or 180/-180.  A small 
        # the longitude grid is therefore adjusted by an epsilon.
        ep = 0.00001
        
        lon_tmp_d = []
        if np.abs(user_lon_0 ) == 180:
            lon_tmp_d.append([ep, 180])
            lon_tmp_d.append([-180, -ep])
        elif user_lon_0 < 0:
            lon_tmp_d.append([-180,180+user_lon_0])
            lon_tmp_d.append([180 + user_lon_0 + ep, 180])
        elif user_lon_0 > 0:
            lon_tmp_d.append([-180+user_lon_0, 180])
            lon_tmp_d.append([-180, -180+user_lon_0-ep])
        elif user_lon_0 == 0:
            # if the map is centered exactly at 0E 
            # cartopy has no problem including -180 and 180 in the longitude
            # arrays
            lon_tmp_d.append([-180, 180])
            
        # loop through different parts of the map to plot (if they exist),
        # do interpolation and plot

        for ki, k in enumerate(lon_tmp_d):
            lon_limits = lon_tmp_d[ki]
            
            new_grid_lon_centers, new_grid_lat_centers,\
            new_grid_lon_edges, new_grid_lat_edges,\
            data_latlon_projection = \
                resample_to_latlon(lons, lats, data,
                                   -90, 90, dy,
                                   lon_limits[0], lon_limits[1], dx,
                                   mapping_method='nearest_neighbor',
                                   radius_of_influence = radius_of_influence)
    
            if plot_type == 'pcolormesh':
                plot_lons = new_grid_lon_edges
                plot_lats = new_grid_lat_edges
            else:
                plot_lons = new_grid_lon_centers
                plot_lats = new_grid_lat_centers
                
            # don't make the colorbar a second time if we've already
            # made it once in the loop
            if cbar != []:
                show_colorbar=False;

            p, gl, cbar = \
                  plot_global(plot_lons,
                              plot_lats,
                              data_latlon_projection,
                              data_epsg_code,
                              cmin, cmax, ax,
                              plot_type = plot_type,
                              cmap = cmap,
                              show_coastline = show_coastline,
                              show_colorbar = show_colorbar,
                              show_land = show_land,
                              show_grid_lines = show_grid_lines,
                              show_grid_labels = show_grid_labels,
                              show_coastline_over_data = show_coastline_over_data,
                              show_land_over_data = show_land_over_data,
                              grid_linewidth = grid_linewidth,
                              grid_linestyle = grid_linestyle,
                              colorbar_label =colorbar_label,
                              less_output=less_output)
    

    return f, ax, p, cbar, new_grid_lon_centers, new_grid_lat_centers,\
        data_latlon_projection, gl
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_pstereo(xx, yy, data,
                 data_epsg_code, \
                 lat_lim,
                 cmin, cmax, ax,
                 plot_type = 'pcolormesh',
                 circle_boundary = False,
                 cmap = None,
                 show_coastline = True,
                 show_colorbar = False,
                 show_land = True,
                 show_grid_lines = True,
                 show_grid_labels = False,
                 show_coastline_over_data = True,
                 show_land_over_data = True,
                 grid_linewidth = 1,
                 grid_linestyle = '--',
                 levels = 20,
                 data_zorder = 50,
                 colorbar_label = None,
                 less_output = True):

    # assign cmap default
    if cmap is None:
        cmap, (new_cmin,new_cmax) = assign_colormap(data, cmap)

    if isinstance(ax.projection, ccrs.NorthPolarStereo):
        ax.set_extent([-180, 180, lat_lim, 90], ccrs.PlateCarree())
        if not less_output:
            print('North Polar Projection')
    
    elif isinstance(ax.projection, ccrs.SouthPolarStereo):
        ax.set_extent([-180, 180, -90, lat_lim], ccrs.PlateCarree())
        if not less_output:
            print('South Polar Projection')
    else:
        raise ValueError('ax must be either ccrs.NorthPolarStereo or ccrs.SouthPolarStereo')

    if not less_output:
        print('lat_lim: ',lat_lim)

    # truncate the plot to just a circular shape
    if circle_boundary:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if data_epsg_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        # reproject the data if necessary
        data_crs=ccrs.epsg(data_epsg_code)


    # plot the data on zorder 50
    p=[]
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs,
                          vmin=cmin, vmax=cmax, cmap=cmap, zorder=data_zorder)

    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,
                        vmin=cmin, vmax=cmax, cmap=cmap, zorder=data_zorder)

    elif plot_type == 'points':
        p = ax.plot(xx, yy,  'k.', transform=data_crs,zorder=data_zorder)

    else:
        raise ValueError('plot_type  must be either "pcolormesh", "contourf", or "points"')


    gl, cbar =\
        _add_features_to_axis(ax, p, cmin, cmax,
                              cmap = cmap,
                              show_coastline = show_coastline,
                              show_colorbar = show_colorbar,
                              show_land = show_land,
                              show_grid_lines = show_grid_lines,
                              show_grid_labels = show_grid_labels,
                              show_coastline_over_data = show_coastline_over_data,
                              show_land_over_data = show_land_over_data,
                              grid_linewidth = grid_linewidth,
                              grid_linestyle = grid_linestyle,
                              colorbar_label = colorbar_label)
        

    return p, gl, cbar

#%%

def plot_global(xx,yy, data,
                data_epsg_code,
                cmin, cmax, ax,
                cmap=None,
                plot_type = 'pcolormesh',
                circle_boundary = False,
                show_coastline = True,
                show_colorbar = False,
                show_land = True,
                show_grid_lines = True,
                show_grid_labels = False,
                show_coastline_over_data = True,
                show_land_over_data = True,
                grid_linewidth = 1,
                grid_linestyle = '--',
                levels = 20,
                colorbar_label=None,
                data_zorder = 50,
                less_output=True):

    # assign cmap default
    if cmap is None:
        cmap, (new_cmin,new_cmax) = assign_colormap(data, cmap)

    if data_epsg_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs =ccrs.epsg(data_epsg_code)

    # plot the data on zorder 50
    p=[]
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs, \
                          vmin=cmin, vmax=cmax, cmap=cmap,
                          zorder=data_zorder)

    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,  \
                        vmin=cmin, vmax=cmax, cmap=cmap,
                        zorder=data_zorder)

    elif plot_type == 'points':
        p = ax.plot(xx, yy,  'k.', transform=data_crs,
                    zorder=data_zorder)

    else:
        raise ValueError('plot_type  must be either "pcolormesh", "contourf", or "points"')

    gl = []
    cbar = []
    
    gl, cbar =\
        _add_features_to_axis(ax, p, cmin, cmax,
                              cmap = cmap,
                              show_coastline = show_coastline,
                              show_colorbar = show_colorbar,
                              show_land = show_land,
                              show_grid_lines = show_grid_lines,
                              show_grid_labels = show_grid_labels,
                              show_coastline_over_data = show_coastline_over_data,
                              show_land_over_data = show_land_over_data,
                              grid_linewidth = grid_linewidth,
                              grid_linestyle = grid_linestyle,
                              colorbar_label = colorbar_label)

    return p, gl, cbar

# -----------------------------------------------------------------------------

def _add_features_to_axis(ax, p, cmin, cmax,
                          cmap = 'jet',
                          show_coastline=True,
                          show_colorbar=True,
                          show_land=True, 
                          show_grid_lines=True,
                          show_grid_labels=True,
                          show_coastline_over_data = True,
                          show_land_over_data = True,
                          grid_linewidth = 1,
                          grid_linestyle = '--',
                          colorbar_label = None):
                                     
    
    if show_land:
        if show_land_over_data:
        # place land over the data
            zorder = 75
        else:
        # place land under the data
            zorder = 25
        
        ax.add_feature(cfeature.LAND, zorder=zorder)
          
    if show_coastline:
        # place the coastline over land and over the data (default zorder 50)
        if show_coastline_over_data:
            zorder = 85
        else:
            # place coastline over land but under data
            zorder = 35
        
        ax.coastlines(linewidth=0.8, zorder=zorder)
        
    gl = []
    if show_grid_lines :
        # grid lines go over everything (zorder 110)
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          linewidth=grid_linewidth, 
                          color='black',
                          draw_labels = show_grid_labels,
                          alpha=0.5, 
                          linestyle=grid_linestyle,
                          zorder=110)
        
    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
        #cbar = plt.colorbar(p, ax=ax)
        if type(colorbar_label) is str:
            cbar.set_label(colorbar_label)
        
    return gl, cbar
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
        True = show the grid labels, possibly not supported by all projections. 
               See cartopy documentation for updates.
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
             'Mollweide':ccrs.Mollweide,
             'ortho': ccrs.Orthographic,
             'InterruptedGoodeHomolosine':ccrs.InterruptedGoodeHomolosine
             }

    try:

    # cartopy crs changed the name of ths proj version attribute
    # so we must check both the new and old names
    if hasattr(cartopy._crs, 'PROJ_VERSION'):
       proj_version = cartopy._crs.PROJ_VERSION
    elif hasattr(cartopy._crs, 'PROJ4_VERSION'):
       proj_version = cartopy._crs.PROJ4_VERSION
    else:
       # I can't tell the proj veresion
       proj_version = (0,0,0) 

    # This projection requires proj4 v.>= 5.2.0
    if proj_version >= (5,2,0):
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

    #if (projection_type == 'Mercator') | (projection_type== 'PlateCarree'):
    ##    show_grid_labels = True
    #else:
    ##    show_grid_labels = False

    if not less_output:
        print('Projection type: ', projection_type)

    return ax
