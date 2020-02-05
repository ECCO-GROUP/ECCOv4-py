"""
ECCO v4 Python: tile_plot

This module includes utility routines for plotting fields in the 
llc 13-tile native flat binary layout.  This layout is the default for 
MITgcm input and output for global setups using lat-lon-cap (llc) layout. 
The llc layout is used for ECCO v4. 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py

"""

from __future__ import division, print_function
import numpy as np
import matplotlib.pylab as plt
import xarray as xr
from distutils.util import strtobool
import pyresample as pr
import xmitgcm
import dask

def plot_tile(tile, cmap='jet', show_colorbar=False,  show_cbar_label=False, 
              cbar_label = '', **kwargs):
    """

    Plots a single tile of the lat-lon-cap (LLC) grid
    
    Parameters
    ----------
    tile : ndarray
        a single 2D tile of dimension llc x llc 

    cmap : colormap, optional, default jet
        a colormap for the figure
  
    show_colorbar : boolean, optional, default False
        add a colorbar
        
    show_cbar_label, boolean, 
        boolean, show a label on the colorbar
        Default: False
        
    less_output : boolean, default False
        A debugging flag.  False = less debugging output
                
    cmin/cmax : floats, optional, default calculate using the min/max of the data
        the minimum and maximum values to use for the colormap
        
    fig_num : int, optiona, default -1 (make new figure)
        integer, the figure number to make the plot on.
        Default: make a new figure
        
    Returns
    -------
    f : Figure
        a handle to the figure

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
            print("unrecognized argument ", key )
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
    
    
def plot_tiles(tiles, cmap='jet', 
               layout='llc', rotate_to_latlon=False,
               Arctic_cap_tile_location = 2,
               show_colorbar=False,  
               show_cbar_label=False, 
               show_tile_labels= True,
               cbar_label = '', 
               fig_size = 9,  **kwargs):
    """

    Plots the 13 tiles of the lat-lon-cap (LLC) grid
    
    Parameters
    ----------
    tiles : numpy.ndarray or dask.array.core.Array or xarray.core.dataarray.DataArray
        an array of n=1..13 tiles of dimension n x llc x llc 

            - If *xarray DataArray* or *dask Array* tiles are accessed via *tiles.sel(tile=n)*
            - If *numpy ndarray* tiles are acceed via [tile,:,:] and thus n must be 13.

    cmap : matplotlib.colors.Colormap, optional, default: jet
        a colormap for the figure

    layout : string, optional, default 'llc'
        a code indicating the layout of the tiles

        :llc:    situates tiles in a fan-like manner which conveys how the tiles 
                 are oriented in the model in terms of x an y
    
        :latlon: situates tiles in a more geographically recognizable manner.  
                 Note, this does not rotate tiles 7..12, it just places tiles 
                 7..12 adjacent to tiles 0..5.  To rotate tiles 7..12 
                 specifiy *rotate_to_latlon* as True
                     
    rotate_to_latlon : boolean, default False
        rotate tiles 7..12 so that columns correspond with
        longitude and rows correspond to latitude.  Note, this rotates
        vector fields (vectors positive in x in tiles 7..12 will be -y 
        after rotation).  

    Arctic_cap_tile_location : int, default 2
        integer, which lat-lon tile to place the Arctic tile over. can be 
        2, 5, 7 or 10.
        
    show_colorbar : boolean, optional, default False
        add a colorbar
        
    show_cbar_label : boolean, optional, default False
        add a label on the colorbar
        
    show_tile_labels : boolean, optional, default True
        show tiles numbers in subplot titles
        
    cbar_label : str, optional, default '' (empty string)
        the label to use for the colorbar
      
    less_output : boolean, optional, default False
        A debugging flag.  False = less debugging output
                
    cmin/cmax : floats, optional, default calculate using the min/max of the data
        the minimum and maximum values to use for the colormap
        
    fig_size : float, optional, default 9 inches
        size of the figure in inches
        
    fig_num : int, optional, default none
        the figure number to make the plot in.  By default make a new figure.
        
    Returns
    -------
    f : Figure
        

    """

    # by default take the min and max of the values
    
    if isinstance(tiles, dask.array.core.Array):
        tiles = np.asarray(tiles.squeeze())
        
    
    if type(tiles) == np.ndarray:
        cmin = np.nanmin(tiles)
        cmax = np.nanmax(tiles)
                    
    elif isinstance(tiles, dask.array.core.Array) or \
         isinstance(tiles, xr.core.dataarray.DataArray):
        cmin = np.nanmin(tiles.values)
        cmax = np.nanmax(tiles.values)
        
             
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
            print("unrecognized argument ", key)


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
        tile_order = np.array([-1, -1, 10, 11, 12, \
                               -1,  6,  7,  8,  9, \
                                2,  5, -1, -1, -1, \
                                1,  4, -1, -1, -1, \
                                0,  3, -1, -1, -1])

    elif layout == 'latlon':
        if fig_num > 0:
            f, axarr = plt.subplots(4, 4, num=fig_num)
        else:
            f, axarr = plt.subplots(4, 4)
                  
        # plotting of the tiles happens in a 4x4 grid
        # which tile to plot for any one of the 16 spots is indicated with a list
        # a value of negative one means do not plot anything in that spot.
        # the top row will have the Arctic tile.  You can choose where the 
        # Arctic tile goes.  By default it goes in the second column.

        if Arctic_cap_tile_location not in [2,5,7,10]:
            print('Arctic Cap Alignment is not one of 2,5,7,10, using 2')
            Arctic_cap_tile_location  = 2    
            
        if  Arctic_cap_tile_location == 2: # plot in 1st position, column 1
            tile_order_top_row = [6, -1, -1, -1]
        elif Arctic_cap_tile_location == 5:
            tile_order_top_row = [-1, 6, -1, -1]
        elif Arctic_cap_tile_location == 7:# plot in 3rd position, column 3
            tile_order_top_row = [-1, -1, 6, -1]
        elif Arctic_cap_tile_location == 10:# plot in 4th position, column 4
            tile_order_top_row = [-1, -1, -1, 6]
            
        # the order of the rest of the tile is fixed.  four columns each with 
        # three rows.
        tile_order_bottom_rows =[2, 5, 7, 10, \
                                 1, 4, 8, 11, \
                                 0, 3, 9, 12]
        
        # these are lists so to combine tile_orde_first and tile_order_rest 
        # you just add them in python (wierd).  If these were numpy arrays 
        # one would use np.concatenate()
        tile_order = tile_order_top_row + tile_order_bottom_rows

    #%%
    #print(fac1, fac2)
    f.set_size_inches(fac1*fig_size, fig_size*fac2)

    if show_tile_labels==False:
        f.subplots_adjust(wspace=0, hspace=0)
    
    #print(f.get_size_inches())

    #if tile_start_index == -1 and type(tiles) == xr.core.dataarray.DataArray:
    #    min_tile_num = np.min(tiles.tile.values)
    #    max_tile_num = np.max(tiles.tile.values)

    #    print (min_tile_num, max_tile_num)
        
        #if min_tile_num == 0 and max_tile_num == 12:
        #    tile_start_index = 0
        #elif min_tile_num == 1 and max_tile_num == 13:
        #    tile_start_index = 1
        #else:
        #    print ('I cannot guess which index you use for the first tile, 0')
        #    print ('or 1, using 0')
        #    tile_start_index = 0
        
    #    print ('ts1 = ',  tile_start_index)

    
    
    # loop through the axes array and plot tiles where tile_order != -1
    cur_arr = np.zeros((360,360)) 
    for i, ax in enumerate(axarr.ravel()):
        ax.axis('off')

        cur_tile_num = tile_order[i]
        
        have_tile = False
        #print i, cur_tile_num
        if cur_tile_num >= 0:
            if type(tiles) == np.ndarray:
                have_tile = True
                cur_tile = tiles[cur_tile_num ]
                    
            elif isinstance(tiles, dask.array.core.Array) or \
                 isinstance(tiles, xr.core.dataarray.DataArray):
                
                if cur_tile_num in tiles.tile :
                    have_tile = True
                    cur_tile = tiles.sel(tile=cur_tile_num)
                
            if have_tile:
                if (layout == 'latlon' and rotate_to_latlon and cur_tile_num == 6):
                    if Arctic_cap_tile_location == 2:
                        cur_tile = np.rot90(cur_tile,-1)
                    elif Arctic_cap_tile_location == 7:
                        cur_tile = np.rot90(cur_tile,-3)
                    elif Arctic_cap_tile_location == 10:
                        cur_tile = np.rot90(cur_tile,2)

                if (layout == 'latlon' and rotate_to_latlon and 
                    cur_tile_num > 6):
                    
                    cur_tile = np.rot90(cur_tile)

                
                    
                im=ax.imshow(cur_tile, vmin=cmin, vmax=cmax, cmap=cmap, 
                             origin='lower')
            
            colnum = 4-1-int(i/4)
            rownum = i%4 
            rownump1 = int(rownum + 1)
            colnump1 = int(colnum + 1)
            print(type(rownum), type(colnum), type(rownump1))
            #cur_arr[colnum*90:colnump1*90, rownum*90:rownump1*90] = cur_tile
            print('i=',i,rownum, colnum)
            cur_arr[colnum*90:colnump1*90, rownum*90:rownump1*90] = cur_tile
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

    return f, cur_arr
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
def unique_color(n=1):
    """

    Returns one of 14 unique colors.  
    see https://xkcd.com/color/rgb/
    and https://matplotlib.org/tutorials/colors/colors.html
    
    Parameters
    ----------
    n : int, optional, default 1
        which unique color do you want [1..13]
        if n is not in 1..13, return mint
        
    Returns
    -------
    c : matplotlib.colors.Colormap
        one of 13 unique colors
        
 
    """
 
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
