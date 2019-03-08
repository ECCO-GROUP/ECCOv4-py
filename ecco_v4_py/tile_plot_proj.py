#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:11:15 2017

@author: ifenty
"""
from __future__ import division,print_function
import numpy as np
import matplotlib.pylab as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ecco_v4_py.resample_to_latlon as resample_to_latlon

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
def plot_proj_to_latlon_grid(lons, lats, data, 
                             projection_type = 'robin', 
                             plot_type = 'pcolormesh', 
                             user_lon_0 = -66,
                             lat_lim = 50, 
                             levels = 20, 
                             cmap='jet', 
                             dx=.25, 
                             dy=.25,
                             show_colorbar = False, 
                             show_grid_lines = True,
                             show_grid_labels = True,
                             **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # default projection type = robinson
    # default central longitude = 60W
    # default no colorbar, no grid labels, no grid lines.
    # default color limits take the min and max of the values
    # default plot_type is pcolormesh.
    # default lat/lon spacing in lat/lon grid is 0.25 degrees
    # default number of levels for contourf is 20 (levels)
    # default latitude limit for polar stereographic plots is 50N (lat_lim)
    # default colormap is 'jet'

    #%%    
    cmin = np.nanmin(data)
    cmax = np.nanmax(data)

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
        
    elif user_lon_0 == 180 or user_lon_0 == -180:
        A_left_limit = -180
        A_right_limit = 0
        B_left_limit =  0
        B_right_limit = 180
        center_lon = 0
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

    # Make projection axis
    if projection_type == 'Mercator':
        ax = plt.axes(projection =  ccrs.Mercator(central_longitude=user_lon_0))

    elif projection_type == 'PlateCaree':
        ax = plt.axes(projection = ccrs.PlateCarree(central_longitude=user_lon_0))

    elif projection_type == 'cyl':
        ax = plt.axes(projection = ccrs.LambertCylindrical(central_longitude=user_lon_0))
        print ('Cannot label gridlines on a LambertCylindrical plot.  Only PlateCarree and Mercator plots are currently supported.')        
        show_grid_labels = False

    elif projection_type == 'robin':    
        ax = plt.axes(projection = ccrs.Robinson(central_longitude=user_lon_0))
        show_grid_labels=False
        print ('Cannot label gridlines on a Robinson plot.  Only PlateCarree and Mercator plots are currently supported.')
        show_grid_labels = False

    elif projection_type == 'ortho':
        ax = plt.axes(projection =  ccrs.Orthographic(central_longitude=user_lon_0))
        print ('Cannot label gridlines on a Orthographic plot.  Only PlateCarree and Mercator plots are currently supported.')        
        show_grid_labels = False

    elif projection_type == 'stereo':    
        if lat_lim > 0:
            ax = plt.axes(projection =ccrs.NorthPolarStereo())
        else:
            ax = plt.axes(projection =ccrs.SouthPolarStereo())

        print ('Cannot label gridlines on a polar stereographic plot.  Only PlateCarree and Mercator plots are currently supported.')            
        show_grid_labels = False

    elif projection_type == 'InterruptedGoodeHomolosine':
        print ('Cannot label gridlines on a InterruptedGoodeHomolosine plot.  Only PlateCarree and Mercator plots are currently supported.')            
        
        ax = plt.axes(projection = ccrs.InterruptedGoodeHomolosine(central_longitude=user_lon_0))
        show_grid_labels = False
        
    else:
        raise ValueError('projection type must be either "Mercator", "PlateCaree",  "cyl", "robin", "ortho", or "stereo"')

    print ('projection type ', projection_type)
    

    #%%
    # loop through different parts of the map to plot (if they exist), 
    # do interpolation and plot
    f = plt.gcf()
    print(len(lon_tmp_d))
    for key, lon_tmp in lon_tmp_d.iteritems():

        new_grid_lon, new_grid_lat, data_latlon_projection = \
            resample_to_latlon(lons, lats, data, 
                               -89.5, 89.5, dy,
                               lon_tmp[0], lon_tmp[1], dx, 
                               mapping_method='nearest_neighbor')
            
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
                             circle_boundary=True,
                             cmap=cmap, 
                             show_grid_lines=False)

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
                            show_grid_labels = False)
                    
        if show_grid_lines :
            ax.gridlines(crs=ccrs.PlateCarree(), 
                                  linewidth=1, color='black', 
                                  alpha=0.5, linestyle='--', 
                                  draw_labels = show_grid_labels)
        
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)        
    
    #%%
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)

    ax= plt.gca()

    #%%
    return f, ax, p, cbar
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    

def plot_pstereo(xx,yy, data, 
                 data_projection_code, \
                 lat_lim, 
                 cmin, cmax, ax, 
                 plot_type = 'pcolormesh', 
                 show_colorbar=False, 
                 circle_boundary = False, 
                 cmap='jet', 
                 show_grid_lines=False,
                 levels = 20):

                            
    if isinstance(ax.projection, ccrs.NorthPolarStereo):
        ax.set_extent([-180, 180, lat_lim, 90], ccrs.PlateCarree())
        print('north')
    elif isinstance(ax.projection, ccrs.SouthPolarStereo):
        ax.set_extent([-180, 180, -90, lat_lim], ccrs.PlateCarree())
        print('south')
    else:
        raise ValueError('ax must be either ccrs.NorthPolarStereo or ccrs.SouthPolarStereo')

    print(lat_lim)
    
    if circle_boundary:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    if show_grid_lines :
        gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                          linewidth=1, color='black', 
                          alpha=0.5, linestyle='--')
    else:
        gl = []

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        # reproject the data if necessary
        data_crs=ccrs.epsg(data_projection_code)
    

    p=[]    
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs, \
                          vmin=cmin, vmax=cmax, cmap=cmap)

    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,  \
                 vmin=cmin, vmax=cmax, cmap=cmap)

    else:
        raise ValueError('plot_type  must be either "pcolormesh" or "contourf"')

         
    ax.add_feature(cfeature.LAND)
    ax.coastlines('110m', linewidth=0.8)

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
                cmap='jet', 
                show_grid_lines = True,
                show_grid_labels = True,
                levels=20):

    if show_grid_lines :
        gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                          linewidth=1, color='black', 
                          draw_labels = show_grid_labels,
                          alpha=0.5, linestyle='--')
    else:
        gl = []
        
    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs =ccrs.epsg(data_projection_code)
        
    if plot_type == 'pcolormesh':
        p = ax.pcolormesh(xx, yy, data, transform=data_crs, 
                          vmin=cmin, vmax=cmax, cmap=cmap)
    elif plot_type =='contourf':
        p = ax.contourf(xx, yy, data, levels, transform=data_crs,
                        vmin=cmin, vmax=cmax, cmap=cmap)
    else:
        raise ValueError('plot_type  must be either "pcolormesh" or "contourf"') 
                         
    ax.coastlines('110m', linewidth=0.8)
    ax.add_feature(cfeature.LAND)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
    
    return p, gl, cbar
