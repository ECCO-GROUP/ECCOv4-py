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
from distutils.util import strtobool
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

    # layout:
    #     'llc' : 
    #     'latlon'
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

    # by default use jet colormap
    user_cmap = 'jet'
    
    # by default do not show a colorbar 
    show_colorbar = False

    # by default the colorbar has no label
    show_cbar_label = False

    # by default the layout is the original llc 

    layout = 'llc' # 
    
    # by default take the min and max of the values
    cmin = np.nanmin(tiles)
    cmax = np.nanmax(tiles)
    
    tile_labels = True

    fsize = 9  #figure size in inches (h and w)
    
    rotate_to_latlon = False
    
    # Which lat-lon tile to plot the Arctic tile over. 
    # -- can be 3, 6, 8 or 11.
    aca = 3

    # plotting of the tiles happens in a 4x4 grid
    # which tile to plot for any one of the 16 spots is indicated with a list
    # a value of negative one means do not plot anything in that spot.
    # the top row will have the Arctic tile.  You can choose where the 
    # Arctic tile goes.  By default it goes in the second column.
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
    
    for key in kwargs:
        if key == "cbar":
            show_colorbar = kwargs[key]
        elif key == "user_cmap":
            user_cmap = kwargs[key]
        elif key == "cbar_label":
            show_cbar_label = kwargs[key]
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
        elif key == 'Arctic_Align':
            aca = kwargs[key]
        else:
            print "unrecognized argument ", key 

    # see if aca is one of four valid values
    if len(np.intersect1d([3,6,8,11],aca)) > 0:
        # set the location of the Arctic tile.
        if  aca == 3: # plot in 1st position, column 1
            tile_order_top_row = [7, -1, -1, -1]
        elif aca == 6:# plot in 2nd position, column 2
            tile_order_top_row = [-1, 7, -1, -1]
        elif aca == 8:# plot in 3rd position, column 3
            tile_order_top_row = [-1, -1, 7, -1]
        elif aca == 11:# plot in 4th position, column 4
            tile_order_top_row = [-1, -1, -1, 7]
    else:
        # if not, set it to be 6.
        print 'Arctic Cap Alignment is not one of 3, 6, 8, 11, using 3'
        aca  = 3


    if layout == 'llc' and aca != 6:
        print 'Arctic_Align only makes sense with the lat-lon layout'

    if layout == 'llc' and rotate_to_latlon == True:
        print 'note: rotate_to_latlon only applies when layout="latlon" '

    if layout == 'latlon':

        if tile_labels:            
            f, axarr = plt.subplots(4, 4, figsize=(fsize,fsize))
        else:
            fac = 1.1194
            f, axarr = plt.subplots(4, 4, figsize=(fsize*fac,fsize),
                                    gridspec_kw = {'wspace':0, 'hspace':0})

                  
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
                # make sure we have this tile in the array
                if tiles.shape[0] >= cur_tile_num -1:
                    have_tile = True
                    cur_tile = tiles[cur_tile_num -1]
                    
            else:
                # make sure we have this tile in the array
                if cur_tile_num in tiles.tile.values:
                    have_tile = True
                    cur_tile = tiles.sel(tile=cur_tile_num)
            
            if have_tile:
                if cur_tile_num == 7:
                    if aca == 3:
                        cur_tile = np.copy(np.rot90(cur_tile,-1))
                    elif aca == 8:
                        cur_tile = np.copy(np.rot90(cur_tile,-3))
                    elif aca == 11:
                        cur_tile = np.copy(np.rot90(cur_tile,2))

                elif (layout == 'latlon' and rotate_to_latlon and 
                    cur_tile_num > 7):
                    
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
