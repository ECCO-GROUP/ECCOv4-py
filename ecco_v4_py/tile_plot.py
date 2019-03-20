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


#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_tile(tile, cmap='jet', show_colorbar=False,  show_cbar_label=False, 
              cbar_label = '', **kwargs):
    """

    Plots a single tile of the lat-lon-cap (LLC) grid
    
    Parameters
    ----------
    tile
        a single 2D tile of dimension llc x llc 

    cmap
        a colormap for the figure
        Default: 'jet'

    show_colorbar
        boolean, show the colorbar
        Default: False
        
    show_cbar_label
        boolean, show a label on the colorbar
        Default: False
        
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    cmin/cmax
        float(s), the minimum and maximum values to use for the colormap
        No Default
        
    fig_num
        integer, the figure number to make the plot on.
        Default: make a new figure
        
    Returns
    -------
    f
        a reference to the figure
        
        
    If dimensions nl or nk are singular, they are not included 
        as dimensions in data_tiles

    """

    # by default take the min and max of the values
    cmin = np.nanmin(tile)
    cmax = np.nanmax(tile)
    
    fig_num = -1
    #%%
    for key in kwargs:
        if key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        elif key == 'fig_num':
            fig_num = kwargs[key]
        else:
            print "unrecognized argument ", key 
    #%%

    if fig_num > 0:
        f = plt.figure(num = fig_num)
    else:
        f = plt.figure()
        
    plt.imshow(tile, vmin=cmin, vmax=cmax, cmap=cmap, 
               origin='lower')
    
    plt.xlabel('+x -->')
    plt.ylabel('+y -->')
    
    # show the colorbar
    if show_colorbar:
        cbar = plt.colorbar()
        if show_cbar_label:
            cbar.set_label(cbar_label)

    return f
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    
    
def plot_tiles(tiles, cmap='jet', layout='llc', rotate_to_latlon=False,
               Arctic_cap_tile_location = 3,
               show_colorbar=False,  
               show_cbar_label=False, show_tile_labels= True,
               cbar_label = '', fig_size = 9, **kwargs):
    """

    Plots the 13 tiles of the lat-lon-cap (LLC) grid
    
    Parameters
    ----------
    tiles
        an array of 13 tiles of dimension 13 x llc x llc 

    cmap
        a colormap for the figure
        Default: 'jet'

    layout
        string, either 'llc' or 'latlon'.  
        'llc'    shows the tiles situated on the figure in a fan-like manner
                 which tries to convey how the tiles are linked together in
                 the model.  the orientation of the tiles is consistent with
                 how the model sees the tiles in terms of x and y
        'latlon' shows the tiles situated on the figure in a more geographically
                 recognizable manner.  Note, that if the 12 'lat/lon' tiles 
                 haven't been rotated so that columns are roughly longitude and
                 rows are roughly latitude, then the orientation of the tiles
                 will be consisent with how the model see the tiles in terms of
                 x and y.  
    rotate_to_latlon
        boolean, flag to rotate tiles 8-13 so that columns correspond with
        longitude and rows correspond to latitude.  Note, if the tiles are
        a vector field, this rotation will not make any physical sense.
        Default: False

    Arctic_cap_tile_location
        integer, which lat-lon tile to place the Arctic tile over. can be 
        3, 6, 8 or 11.
        Default: 3
        
    show_colorbar
        boolean, show the colorbar
        Default: False
        
    show_cbar_label
        boolean, show a label on the colorbar
        Default: False
        
    show_tile_labels 
        boolean, show tiles numbers as titles
        Default: True
        
    less_output : boolean
        A debugging flag.  False = less debugging output
        Default: False
        
    cmin/cmax
        float(s), the minimum and maximum values to use for the colormap
        No Default
        
    fig_size
        float, size of the figure in inches
        Default: 9
        
    fig_num
        integer, the figure number to make the plot on.
        Default: make a new figure
        
    Returns
    -------
    f
        a reference to the figure
        
        
    If dimensions nl or nk are singular, they are not included 
        as dimensions in data_tiles

    """

    # by default take the min and max of the values
    cmin = np.nanmin(tiles)
    cmax = np.nanmax(tiles)
    
    fig_num = -1
    #%%
    for key in kwargs:
        if key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        elif key == 'fig_num':
            fig_num = kwargs[key]
        else:
            print "unrecognized argument ", key 

    # plotting of the tiles happens in a 4x4 grid
    # which tile to plot for any one of the 16 spots is indicated with a list
    # a value of negative one means do not plot anything in that spot.
    # the top row will have the Arctic tile.  You can choose where the 
    # Arctic tile goes.  By default it goes in the second column.
    tile_order_top_row = [-1, 7, -1, -1]


    if len(np.intersect1d([3,6,8,11],Arctic_cap_tile_location)) > 0:
        # set the location of the Arctic tile.
        if  Arctic_cap_tile_location == 3: # plot in 1st position, column 1
            tile_order_top_row = [7, -1, -1, -1]
        elif Arctic_cap_tile_location == 6:# plot in 2nd position, column 2
            tile_order_top_row = [-1, 7, -1, -1]
        elif Arctic_cap_tile_location == 8:# plot in 3rd position, column 3
            tile_order_top_row = [-1, -1, 7, -1]
        elif Arctic_cap_tile_location == 11:# plot in 4th position, column 4
            tile_order_top_row = [-1, -1, -1, 7]
    else:
        # if not, set it to be 6.
        print 'Arctic Cap Alignment is not one of 3, 6, 8, 11, using 3'
        Arctic_cap_tile_location  = 3


    #if layout == 'llc' and aca != 6:
    #    print 'Arctic_Align only makes sense with the lat-lon layout'

    fac1 = 1; fac2=1

    if show_tile_labels and show_colorbar:
        fac2 = 1.15

    if show_tile_labels==False:
        if show_colorbar:
            fac2 =  0.8766666666666666
        else:
            fac2 = 9.06/9
        
    if layout == 'llc' :
        if fig_num > 0:
            f, axarr = plt.subplots(5, 5, num=fig_num)
        else:
            f, axarr = plt.subplots(5, 5)
            
        # plotting of the tiles happens in a 5x5 grid
        # which tile to plot for any one of the 25 spots is indicated with a list
        # a value of negative one means do not plot anything in that spot.
        tile_order = np.array([-1, -1, 11, 12, 13, \
                      -1, 7, 8, 9, 10, \
                      3, 6, -1, -1, -1, \
                      2, 5, -1, -1, -1, \
                      1, 4, -1, -1, -1])

    elif layout == 'latlon':
        if fig_num > 0:
            f, axarr = plt.subplots(4, 4, num=fig_num)
        else:
            f, axarr = plt.subplots(4, 4)
                  
        # the order of the rest of the tile is fixed.  four columns each with 
        # three rows.
        tile_order_bottom_rows =[3, 6, 8, 11,
                          2, 5, 9, 12, \
                          1, 4, 10, 13]
        
        # these are lists so to combine tile_orde_first and tile_order_rest 
        # you just add them in python (wierd).  If these were numpy arrays 
        # one would use np.concatenate()
        tile_order = tile_order_top_row + tile_order_bottom_rows

    print fac1, fac2
    f.set_size_inches(fac1*fig_size, fig_size*fac2)

    if show_tile_labels==False:
        f.subplots_adjust(wspace=0, hspace=0)
    
    print f.get_size_inches()

    
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
                if (layout == 'latlon' and rotate_to_latlon and cur_tile_num == 7):
                    if Arctic_cap_tile_location == 3:
                        cur_tile = np.rot90(cur_tile,-1)
                        print 'here'
                    elif Arctic_cap_tile_location == 8:
                        cur_tile = np.rot90(cur_tile,-3)
                    elif Arctic_cap_tile_location == 11:
                        cur_tile = np.rot90(cur_tile,2)

                if (layout == 'latlon' and rotate_to_latlon and 
                    cur_tile_num > 7):
                    
                    cur_tile = np.rot90(cur_tile)
                
                im=ax.imshow(cur_tile, vmin=cmin, vmax=cmax, cmap=cmap, 
                             origin='lower')
            
    
            ax.set_aspect('equal')
            ax.axis('on')
            if show_tile_labels:
                ax.set_title('Tile ' + str(cur_tile_num))
                
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    # show the colorbar
    if show_colorbar:
        if show_tile_labels:
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
