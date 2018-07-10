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


def plot_tile(tile, cmap='jet', **kwargs):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # shows a single llc tile.  
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    
    # by default do not show a colorbar 
    show_colorbar = False

    # by default the colorbar has no label
    show_cbar_label = False

    # by default take the min and max of the values
    cmin = np.nanmin(tile)
    cmax = np.nanmax(tile)
    
    #%%
    for key in kwargs:
        if key == "cbar":
            show_colorbar = kwargs[key]
        elif key == "cbar_label":
            cbar_label = kwargs[key]
            show_cbar_label = True
        elif key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        else:
            print "unrecognized argument ", key 
    #%%
       
    plt.imshow(tile, vmin=cmin, vmax=cmax, cmap=cmap, 
               origin='lower')
    
    plt.xlabel('+x -->')
    plt.ylabel('+y -->')
    
    # show the colorbar
    if show_colorbar:
        cbar = plt.colorbar()
        if show_cbar_label:
            cbar.set_label(cbar_label)

    plt.show()
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
    
def plot_tiles(tiles,  **kwargs):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # this routine plots the 13 llc faces in either the original 'llc' layout
    # or the quasi lat-lon  layout
    # cmin and cmax are the color minimum and maximum
    # max.  'tiles' is a DataArray of a single 2D variable
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    # by default use jet colormap
    user_cmap = 'jet'
    
    # by default do not show a colorbar 
    show_colorbar = False

    # by default the colorbar has no label
    show_cbar_label = False

    # by default the layout is the original llc 
    layout = 'llc'
    
    # by default take the min and max of the values
    cmin = np.nanmin(tiles)
    cmax = np.nanmax(tiles)
    
    tile_labels = True

    fsize = 9  #figure size in inches (h and w)
    
    rotate_to_latlon = False
    
    for key in kwargs:
        if key == "cbar":
            show_colorbar = kwargs[key]
        elif key == "user_cmap":
            user_cmap = kwargs[key]
        elif key == "cbar_label":
            cbar_label = kwargs[key]
            show_cbar_label = True
        elif key == "layout":
            layout = kwargs[key]
        elif key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        elif key == "tile_labels":
            tile_labels = kwargs[key]
        elif key == "fsize":
            fsize = kwargs[key]
        elif key == "rotate_to_latlon":
            rotate_to_latlon = kwargs[key]
        else:
            print "unrecognized argument ", key 
               

    if layout == 'latlon':

        if tile_labels:            
            f, axarr = plt.subplots(4, 4, figsize=(fsize,fsize))
        else:
            fac = 1.1194
            f, axarr = plt.subplots(4, 4, figsize=(fsize*fac,fsize),
                                    gridspec_kw = {'wspace':0, 'hspace':0})

        # plotting of the tiles happens in a 4x4 grid
        # which tile to plot for any one of the 16 spots is indicated with a list
        # a value of negative one means do not plot anything in that spot.
        # the top row will have the Arctic tile.  Where we put the Arctic tile
        # depends on which other tile it is aligned with.  By default, it is
        # aligned with tile 6, which is the second column.  
        tile_order_top_row = [-1, 7, -1, -1]
        
        if type(tiles) == np.ndarray:
            pass
        else:
            # we were sent a Dataset or DataArray   
            # if we have defined the attribute 'Arctic_Align' then perhaps
            # the Arctic cap is algined with another tile and therefore it's location
            # in the figure may be different changed.
            if 'Arctic_Align' in tiles.attrs:
                aca = tiles.attrs['Arctic_Align']
                #print 'Arctic Cap Alignment match with tile: ', aca
                
                if  aca == 3: # plot in 1st position, column 1
                    tile_order_top_row = [7, -1, -1, -1]
                elif aca == 6:# plot in 2nd position, column 2
                    tile_order_top_row = [-1, 7, -1, -1]
                elif aca == 8:# plot in 3rd position, column 3
                    tile_order_top_row = [-1, -1, 7, -1]
                elif aca == 11:# plot in 4th position, column 4
                    tile_order_top_row = [-1, -1, -1, 7]
                else:
                    print 'Arctic Cap Alignment is not one of 3, 6, 8, 11.'
                        
        # the order of the rest of the tile is fixed.  four columns each with 
        # three rows.
        tile_order_bottom_rows =[3, 6, 8, 11,
                          2, 5, 9, 12, \
                          1, 4, 10, 13]
        
        # these are lists so to combine tile_orde_first and tile_order_rest 
        # you just add them in python (wierd).  If these were numpy arrays 
        # one would use np.concatenate()
        tile_order = tile_order_top_row + tile_order_bottom_rows
    
    elif layout == 'llc':
        if tile_labels:            
            f, axarr = plt.subplots(5,5, figsize=(fsize,fsize))
        else:
            fac = 1.1194
            f, axarr = plt.subplots(5, 5, figsize=(fsize*fac,fsize),
                                    gridspec_kw = {'wspace':0, 'hspace':0})
            
    
        # plotting of the tiles happens in a 5x5 grid
        # which tile to plot for any one of the 25 spots is indicated with a list
        # a value of negative one means do not plot anything in that spot.
        tile_order = np.array([-1, -1, 11, 12, 13, \
                      -1, 7, 8, 9, 10, \
                      3, 6, -1, -1, -1, \
                      2, 5, -1, -1, -1, \
                      1, 4, -1, -1, -1])
        


    # loop through the axes array and plot tiles where tile_order != -1
    for i, ax in enumerate(axarr.ravel()):
        ax.axis('off')

        cur_tile_num = tile_order[i]
        
        have_tile = False
        #print i, cur_tile_num
        if cur_tile_num > 0:
            if type(tiles) == np.ndarray:
                #print 'we have an ndarray'
                # make sure we have this tile in the array
                if tiles.shape[0] >= cur_tile_num -1:
                    have_tile = True
                    cur_tile = tiles[cur_tile_num -1]
                    
            else:
                # make sure we have this tile in the array
                #print ' we have a DataArray'
                #print tiles.tile
                if cur_tile_num in tiles.tile.values:
                    have_tile = True
                    cur_tile = tiles.sel(tile=cur_tile_num)
                    
            #print cur_tile_num, have_tile
            if have_tile:
                if (layout == 'latlon' and rotate_to_latlon and 
                    cur_tile_num >7):
                    
                    cur_tile = np.copy(np.rot90(cur_tile))
                
                im=ax.imshow(cur_tile, vmin=cmin, vmax=cmax, cmap=user_cmap, 
                             origin='lower')
    
            ax.set_aspect('equal')
            ax.axis('on')
            if tile_labels:
                ax.set_title('Tile ' + str(cur_tile_num))
                
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # show the colorbar
    if show_colorbar:
        if tile_labels:
            f.subplots_adjust(left=None, bottom=None, right=0.8)
        else:
            f.subplots_adjust(right=0.8, left=None, bottom=None,
                              top=None, wspace=0, hspace=0)
            
        #[left, bottom, width, height]
        h=.6;w=.025
        cbar_ax = f.add_axes([0.85, (1-h)/2, w, h])
        cbar = f.colorbar(im, cax=cbar_ax)#, format='%.0e')        
        if show_cbar_label:
            cbar.set_label(cbar_label)

    return f
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

def plot_tiles_proj(lons, lats, data, 
                    user_lat_0 = 45, 
                    projection_type = 'robin', 
                    plot_type = 'pcolor', 
                    user_lon_0 = 0, 
                    background_type = 'fc', 
                    show_cbar_label = False, 
                    show_colorbar = False, 
                    bound_lat = 50, 
                    num_levels = 20, 
                    cmap='jet', 
                    map_resolution='c',
                    dx=.25, 
                    dy=.25, **kwargs):
    
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

    
    # create the basemap object, 'map'
    if projection_type == 'cyl':
        map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
                llcrnrlon=A_left_limit, urcrnrlon=B_right_limit, 
                resolution=map_resolution)
    
    elif projection_type == 'robin':    
        map = Basemap(projection='robin',lon_0=center_lon, 
                      resolution=map_resolution)

    elif projection_type == 'ortho':
        map = Basemap(projection='ortho',lat_0=user_lat_0,lon_0=user_lon_0,
                      resolution=map_resolution)
    elif projection_type == 'stereo':    
        if bound_lat > 0:
            map = Basemap(projection='npstere', boundinglat = bound_lat,
                          lon_0=user_lon_0, resolution=map_resolution)
        else:
            map = Basemap(projection='spstere', boundinglat = bound_lat,
                          lon_0=user_lon_0, resolution=map_resolution)
    else:
        raise ValueError('projection type must be either "cyl", "robin", or "stereo"')
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
    if projection_type == 'robin':      
        map.drawmeridians(np.arange(0,360,30))
        map.drawparallels(np.arange(-90,90,30))
    elif projection_type == 'stereo':      
        map.drawmeridians(np.arange(0,360,30))
        map.drawparallels(np.arange(-90,90,10)) 
    elif projection_type == 'cyl':
        # labels = [left,right,top,bottom]
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




    
    
def unique_color(n):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # returns one of 13 unique colors.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if n == 1:
        c='xkcd:red'
    elif n== 2:
        c='xkcd:green'
    elif n== 3:
        c='xkcd:yellow'
    elif n== 4:
        c='xkcd:blue'
    elif n== 5:
        c='xkcd:orange'
    elif n== 6:
        c='xkcd:purple'
    elif n== 7:
        c='xkcd:cyan'
    elif n== 8:
        c='xkcd:magenta'
    elif n== 9:
        c='xkcd:lime green'
    elif n== 10:
        c='xkcd:candy pink'
    elif n== 11:
        c='xkcd:teal'
    elif n== 12:
        c='xkcd:lavender'
    elif n== 13:
        c='xkcd:brown'
    else:
        c='xkcd:mint'

    return c
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
