#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:01:48 2018

@author: ifenty
"""
from __future__ import division
import numpy as np
import matplotlib.pylab as plt
import xarray as xr

# import all function from the 'mpl_toolkits.basemap' module available with the 
# prefix 'Basemap'
from mpl_toolkits.basemap import Basemap


def plot_latlon_interp_proj(lons, lats, data,  **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # this routine plots the 0.5 degree interpolated fields
    # in either cylindrical or robinson projections
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if type(lons) == xr.DataArray:
        lons = lons.values
    elif type(lons) != np.ndarray:
        print 'lons must be either a DataArray or ndarray type \n'
        print 'found type ', type(lons)

    if type(lats) == xr.DataArray:
        lats = lats.values
    elif type(lats) != np.ndarray:
        print 'lats must be either a DataArray or ndarray type \n'
        print 'found type ', type(lats)
        
    if type(data) == xr.DataArray:
        data = data.values
    elif type(data) != np.ndarray:
        print 'data must be either a DataArray or ndarray type \n'
        print 'found type ', type(data)

    # default projection type
    projection_type = 'robin'
    
    # by default the left most longitude in the global map is -180E.
    user_lon_0 = 0

    # by default use jet colormap
    user_cmap = 'jet'
    
    # by default do not show a colorbar 
    show_colorbar = False

    # by default the colorbar has no label
    show_cbar_label = False

    # by default take the min and max of the values
    cmin = np.nanmin(data)
    cmax = np.nanmax(data)
   
    # by default the plot_type is pcolor.
    plot_type = 'pcolor'
    
    # the default number of levels for contourf, 
    num_levels = 20
    
    # default background is to fill continents with gray color
    background_type = 'fc'
        
    #%%
    for key in kwargs:
        if key == "lon_0" :
            user_lon_0 = kwargs[key]
        elif key == "cbar":
            show_colorbar = kwargs[key]
        elif key == "user_cmap":
            user_cmap = kwargs[key]
        elif key == "cbar_label":
            cbar_label = kwargs[key]
            show_cbar_label = True
        elif key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        elif key == "plot_type":
            plot_type =  kwargs[key]
        elif key == "num_levels":
            num_levels =  kwargs[key]
        elif key == "projection_type":
            projection_type = kwargs[key]
        elif key == "background_type":
            background_type = kwargs[key]
        else:
            print "unrecognized argument ", key     

    #%%
    # To avoid plotting problems around the date line, lon=180E, -180W 
    # I take the approach of plotting the field in two parts, A and B.  
    # Typically part 'A' spans from starting longitude to 180E while part 'B' 
    # spans the from 180E to 360E + starting longitude.  If the starting 
    # longitudes or 0 or 180 special case.
    
    two_parts = False
    if user_lon_0 == 180 or user_lon_0 == -180:
        A_left_limit = -180
        A_right_limit = 0
        B_left_limit =  0
        B_right_limit = 180
        center_lon = 0

    elif user_lon_0 > -180 and user_lon_0 < 180:
        A_left_limit = user_lon_0
        A_right_limit = 180
        B_left_limit =  180
        B_right_limit = 360+user_lon_0
        center_lon = A_left_limit + 180
        two_parts = True

        tmp = lons[0,:]
        idx_AL = np.searchsorted(tmp,A_left_limit, side="left")
        idx_AR = np.searchsorted(tmp,A_right_limit, side="left")
    
        tmpB = tmp+360;
        idx_BL = np.searchsorted(tmpB,B_left_limit, side="left")
        idx_BR = np.searchsorted(tmpB,B_right_limit, side="left")

        print idx_AL, idx_AR, idx_BL, idx_BR
    else:
        print 'invalid starting longitude'
        #return

    
    #%%
    # create the basemap object, 'map'
    if projection_type == 'cyl':
        map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=A_left_limit, urcrnrlon=B_right_limit, 
                resolution='c')
    
    elif projection_type == 'robin':    
        map = Basemap(projection='robin',lon_0=center_lon, resolution='c')

    else:
        print 'projection type must be either "cyl" or "robin" '  
        print 'found ', projection_type
        #return
    
    #%%
    # get a reference to the current figure (or make a figure if none exists)
    f = plt.gcf()

    if background_type == 'bm':
        map.bluemarble()
        print 'blue marble'
    elif background_type == 'sr':
        map.shadedrelief()
        print 'shaded relief'        
    elif background_type == 'fc':
        map.fillcontinents(color='lightgray',lake_color='lightgray')  
        print 'gray background'                

    x,y = map(lons, lats) 
    xb,yb = map(lons+360, lats) 
    
    #%%
    if plot_type == 'pcolor':
        # plot using pcolor 
        if two_parts:
            im=map.pcolor(x[:,idx_AL:idx_AR],y[:,idx_AL:idx_AR], 
                          data[:,idx_AL:idx_AR], 
                          vmin=cmin, vmax=cmax, cmap=user_cmap)
            im=map.pcolor(xb[:,idx_BL:idx_BR],y[:,idx_BL:idx_BR], 
                          data[:,idx_BL:idx_BR], 
                          vmin=cmin, vmax=cmax, cmap=user_cmap)
        else:
            im=map.pcolor(x,y,data,
                          vmin=cmin, vmax=cmax, cmap=user_cmap)

    elif plot_type == 'contourf':
        # create a set of contours spanning from cmin to cmax over
        # num_levels intervals
        contour_levels = np.linspace(cmin, cmax, num_levels)
        
        # plot using contourf
        if two_parts:
            im=map.contourf(x[:,idx_AL:idx_AR],y[:,idx_AL:idx_AR], 
                            data[:,idx_AL:idx_AR], 
                            num_levels,
                            vmin=cmin, vmax=cmax, cmap=user_cmap, 
                            levels=contour_levels, extend="both")
            im=map.contourf(xb[:,idx_BL:idx_BR],y[:,idx_BL:idx_BR], 
                            data[:,idx_BL:idx_BR], 
                            num_levels,
                            vmin=cmin, vmax=cmax, cmap=user_cmap, 
                            levels=contour_levels, extend="both")
        else:
            im=map.contourf(x,y,data,
                            num_levels,
                            vmin=cmin, vmax=cmax, cmap=user_cmap, 
                            levels=contour_levels, extend="both")
    else:
        print 'plot type must be either "pcolor" or "contourf"  '
        print 'found type ', plot_type
        #return
        
    #%%
           
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=1)
    # don't plot lat/lon labels for robinson     projection.
    if projection_type == 'robin':      
        map.drawmeridians(np.arange(0,330,30))
        map.drawparallels(np.arange(-60,61,30))
    else:
        # labels = [left,right,top,bottom]
        map.drawparallels(np.arange(-60,61,30), labels=[True,False,False,False])    
        map.drawmeridians(np.arange(0,301,60),  labels= [False,False, False,True])
    
    #%%
    ax= plt.gca()

    if show_colorbar:
        f=plt.gcf()
        f.subplots_adjust(right=0.8)
        #[left, bottom, width, height]
        h=.6;w=.025
        cbar_ax = f.add_axes([0.85, (1-h)/2, w, h])
        cbar = f.colorbar(im, extend='both', cax=cbar_ax)#, format='%.0e')          

        if show_cbar_label:
            cbar.set_label(cbar_label)
    # set the current axes to be the map, not the colorbar
    plt.sca(ax)

    #%%
    # return a reference to the figure and the map axes
    return f, ax, im
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_latlon_interp(lons, lats, data,  **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    if type(lons) == xr.DataArray:
        lons = lons.values
    elif type(lons) != np.ndarray:
        print 'lons must be either a DataArray or ndarray type \n'
        print 'found type ', type(lons)

    if type(lats) == xr.DataArray:
        lats = lats.values
    elif type(lats) != np.ndarray:
        print 'lats must be either a DataArray or ndarray type \n'
        print 'found type ', type(lats)
        
    if type(data) == xr.DataArray:
        data = data.values
    elif type(data) != np.ndarray:
        print 'data must be either a DataArray or ndarray type \n'
        print 'found type ', type(data)


    # by default use jet colormap
    user_cmap = 'jet'
    
    # by default do not show a colorbar 
    show_colorbar = False

    # by default the colorbar has no label
    show_cbar_label = False

    # by default take the min and max of the values
    cmin = np.nanmin(data)
    cmax = np.nanmax(data)
   
    # by default the plot_type is pcolor.
    plot_type = 'pcolor'
    
    # the default number of levels for contourf, 
    num_levels = 20
    
    #%%
    for key in kwargs:
        if key == "cbar":
            show_colorbar = kwargs[key]
        elif key == "user_cmap":
            user_cmap = kwargs[key]
        elif key == "cbar_label":
            cbar_label = kwargs[key]
            show_cbar_label = True
        elif key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        elif key == "plot_type":
            plot_type =  kwargs[key]
        elif key == "num_levels":
            num_levels =  kwargs[key]
        else:
            print "unrecognized argument ", key     

    #%%
    # get a reference to the current figure (or make a figure if none exists)
    f = plt.gcf()

    #%%
    if plot_type == 'pcolor':
        # plot using pcolor 
        plt.pcolor(lons,lats,data,
                   vmin=cmin, vmax=cmax, 
                   cmap=user_cmap)

    elif plot_type == 'contourf':
        # create a set of contours spanning from cmin to cmax over
        # num_levels intervals
        contour_levels = np.linspace(cmin, cmax, num_levels)
        
        plt.contourf(lons,lats,data,
                     num_levels, 
                     vmin=cmin, vmax=cmax, cmap=user_cmap, 
                     levels=contour_levels, extend="both")
        
    else:
        print 'plot type must be either "pcolor" or "contourf"  '
        print 'found type ', plot_type
        #return
        
  
    #%%
    ax= plt.gca()

    if show_colorbar:
        f=plt.gcf()
        f.subplots_adjust(right=0.8)
        #[left, bottom, width, height]
        h=.6;w=.025
        cbar_ax = f.add_axes([0.85, (1-h)/2, w, h])
        cbar = f.colorbar(im, extend='both', cax=cbar_ax)#, format='%.0e')          

        if show_cbar_label:
            cbar.set_label(cbar_label)
    # set the current axes to be the map, not the colorbar
    plt.sca(ax)

    #%%
    # return a reference to the figure and the map axes
    return f, ax, im
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
