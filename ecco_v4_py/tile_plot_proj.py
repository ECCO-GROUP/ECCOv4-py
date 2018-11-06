#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:11:15 2017

@author: ifenty
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import xarray as xr

# import all function from the 'mpl_toolkits.basemap' module available with the 
# prefix 'Basemap'
from mpl_toolkits.basemap import Basemap

# Performs cartographic transformations and geodetic computations.

# The Proj class can convert from geographic (longitude,latitude) to native 
# map projection (x,y) coordinates and vice versa, or from one map projection 
# coordinate system directly to another.
# https://pypi.python.org/pypi/pyproj?
#
import pyresample as pr

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

def plot_tiles_proj(lons, lats, data, 
                    user_lat_0 = 45, 
                    projection_type = 'robin', 
                    plot_type = 'pcolor', 
                    user_lon_0 = 0,
                    user_width = 5000000,
                    user_height = 4500000,
                    background_type = 'fc', 
                    show_cbar_label = False, 
                    show_colorbar = False, 
                    cbar_label = '',
                    bound_lat = 50, 
                    num_levels = 20, 
                    cmap='jet', 
                    map_resolution='c',
                    dx=.25, 
                    dy=.25,
                    show_grid_lines = True,
                    **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # default projection type
    # by default the left most longitude in the global map is -180E.
    # by default do not show a colorbar 
    # by default the colorbar has no label
    # by default take the min and max of the values
    cmin = np.nanmin(data)
    cmax = np.nanmax(data)
    
    # by default the plot_type is pcolor.
    # the default number of levels for contourf, 
    # default background is to fill continents with gray color
    # default bounding lat for polar stereographic projection is 50 N
    
    
    #%%
    for key in kwargs:
        if key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        else:
            print "unrecognized argument ", key     

    #%%
    if type(lons) == xr.core.dataarray.DataArray:
        lons_1d = lons.values.reshape(np.product(lons.values.shape))
        lats_1d = lats.values.reshape(np.product(lats.values.shape))   

    elif type(lons) == np.ndarray:
        lons_1d = lons.reshape(np.product(lons.shape))
        lats_1d = lats.reshape(np.product(lats.shape) )       
    else:
        print 'lons and lats variable either a DataArray or numpy.ndarray'
        print 'lons found type ', type(lons)
        print 'lats found type ', type(lats)        
        return
    
    if type(data) == xr.core.dataarray.DataArray:
        data = data.values


    elif type(data) != np.ndarray:
        print 'data must be either a DataArray or ndarray type \n'
        print 'found type ', type(data)
        return

    #%%
    # To avoid plotting problems around the date line, lon=180E, -180W 
    # I take the approach of plotting the field in two parts, A and B.  
    # Typically part 'A' spans from starting longitude to 180E while part 'B' 
    # spans the from 180E to 360E + starting longitude.  If the starting 
    # longitudes or 0 or 180 special case.
    if user_lon_0 > -180 and user_lon_0 < 180:
        A_left_limit = user_lon_0
        A_right_limit = 180
        B_left_limit =  180
        B_right_limit = 360+user_lon_0
        center_lon = A_left_limit + 180
        
    elif user_lon_0 == 180 or user_lon_0 == -180:
        A_left_limit = -180
        A_right_limit = 0
        B_left_limit =  0
        B_right_limit = 180
        center_lon = 0
    else:
        print 'invalid starting longitude'
        #return

    # the number of degrees spanned in part A and part B
    num_deg_A =  (A_right_limit - A_left_limit)/dx
    num_deg_B =  (B_right_limit - B_left_limit)/dx

    # We will interpolate the data to the new grid.  Store the longitudes to
    # interpolate to for part A and part B
    lon_tmp_d = dict()
    if num_deg_A > 0:
        lon_tmp_d['A'] = np.linspace(A_left_limit, A_right_limit, num_deg_A)
        
    if num_deg_B > 0:
       lon_tmp_d['B'] = np.linspace(B_left_limit, B_right_limit, num_deg_B)

    print ('projection type ', projection_type)
    # create the basemap object, 'map'
    if projection_type == 'cyl':
        map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=A_left_limit, urcrnrlon=B_right_limit, 
                resolution=map_resolution)
    
    elif projection_type == 'robin':    
        map = Basemap(projection='robin',lon_0=center_lon, 
                      resolution=map_resolution)

    elif projection_type == 'ortho':
        map = Basemap(projection='ortho',lat_0=user_lat_0,lon_0=user_lon_0)

    elif projection_type == 'aeqd':
        map = Basemap(projection='aeqd',lat_0=user_lat_0,lon_0=user_lon_0,
                      resolution=map_resolution, width=user_width,
                      height=user_height)
        
    elif projection_type == 'stereo':    
        if bound_lat > 0:
            map = Basemap(projection='npstere', boundinglat = bound_lat,
                          lon_0=user_lon_0, resolution=map_resolution)
        else:
            map = Basemap(projection='spstere', boundinglat = bound_lat,
                          lon_0=user_lon_0, resolution=map_resolution)
    else:
        raise ValueError('projection type must be either "cyl", "robin", "aqed", or "stereo"')
        print 'found ', projection_type
    
    #%%
    # get a reference to the current figure (or make a figure if none exists)
    if background_type == 'bm':
        map.bluemarble()
        print 'blue marble'
    elif background_type == 'sr':
        map.shadedrelief()
        print 'shaded relief'        
    elif background_type == 'fc':
        map.fillcontinents(color='lightgray',lake_color='lightgray')  
        pass
    
    # prepare for the interpolation or nearest neighbor mapping
    
    # first define the lat lon points of the original data
    orig_grid = pr.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)
    
    # the latitudes to which we will we interpolate
    lat_tmp = np.linspace(-89.5, 89.5, 90/dy)
    
    map.drawcoastlines(linewidth=1)

    # loop through both parts (if they exist), do interpolation and plot
    for key, lon_tmp in lon_tmp_d.iteritems():

        #%%
        new_grid_lon, new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)
    
        
        # define the lat lon points of the two parts. 
        new_grid  = pr.geometry.GridDefinition(lons=new_grid_lon, 
                                               lats=new_grid_lat)
        
        x,y = map(new_grid_lon, new_grid_lat) 
    
        data_latlon_projection = \
            pr.kd_tree.resample_nearest(orig_grid, data, new_grid, 
                                        radius_of_influence=100000, 
                                        fill_value=None) 

        if plot_type == 'pcolor':
            # plot using pcolor 
            im=map.pcolor(x,y, data_latlon_projection, 
                          vmin=cmin, vmax=cmax, cmap=cmap)

        elif plot_type == 'contourf':
            # create a set of contours spanning from cmin to cmax over
            # num_levels intervals
            contour_levels = np.linspace(cmin, cmax, num_levels)
            
            # plot using contourf
            im=map.contourf(x,y, data_latlon_projection, num_levels,
                         vmin=cmin, vmax=cmax, cmap=cmap, 
                         levels=contour_levels, extend="both")
        else:
            print 'plot type must be either "pcolor" or "contourf"  '
            print 'found type ', plot_type
            #return
        
           
    # draw coastlines, country boundaries, fill continents.
    #map.drawcoastlines(linewidth=1)
    # don't plot lat/lon labels for robinson     projection.

    # labels = [left,right,top,bottom]
    if projection_type == 'robin' and show_grid_lines == True:      
        map.drawmeridians(np.arange(0,360,30))
        map.drawparallels(np.arange(-90,90,30))
    elif projection_type == 'stereo' and show_grid_lines == True:      
        map.drawmeridians(np.arange(0,360,30))
        map.drawparallels(np.arange(-90,90,10)) 
    elif projection_type == 'aeqd' and show_grid_lines == True:
        map.drawmeridians(np.arange(0,360,30))
        map.drawparallels(np.arange(-90,90,10))  
    elif projection_type == 'cyl' and show_grid_lines == True:
        map.drawparallels(np.arange(-90,90,30), labels=[True,False,False,False])    
        map.drawmeridians(np.arange(0,360,60),  labels= [False,False, False,True])
    
    #%%
    ax= plt.gca()
    f = plt.gcf()

    if show_colorbar:
        f.subplots_adjust(right=0.8)
        #[left, bottom, width, height]
        h=.6;w=.025
        cbar_ax = f.add_axes([0.85, (1-h)/2, w, h])
        cbar = f.colorbar(im, extend='both', cax=cbar_ax)#, format='%.0e')          

        if show_cbar_label:
            cbar.set_label(cbar_label)
    # set the current axes to be the map, not the colorbar
    plt.sca(ax)

    # return a reference to the figure and the map axes
    return f, ax, im
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

