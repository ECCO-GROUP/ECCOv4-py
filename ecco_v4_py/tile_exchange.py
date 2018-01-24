#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:11:15 2017

@author: ifenty
"""
from __future__ import division
import numpy as np
import xarray as xr
from copy import deepcopy

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def append_border_to_tile(orig_arr, tile_index, point_type,
                          pad_i, pad_j, llcN, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # This routine appends values to one or more horizontal edges of a llc 
    # tile variable using values from its neighboring tiles.  The orientation
    # of the reference tile (with tile index = 'tile_index') must be in its 
    # original llc tile orientation (passed through kwargs).
    #
    # These appended values can be in one of three locations: 
    # 1) a new row along the "top" of the array 
    # 2) a new column to the "right" of the array
    # 3) a new single value in a "corner (g) point" of the array.
    #
    # After the routine the new array will have dimension depending on where
    # the variable is on the Arakawa-C grid 
    # 1) (nx, ny+1) for appending values from variables on the 'v' points 
    # 2) (nx+1, ny) for appending values from variables on the 'u' points  
    # 3) (nx+1, ny+1) for appending values from variables on the 'g' points
    #
    # Appending new values from adjacent neighbors is permitted for tiles
    # that connect to the "top", "right", and/or "corner (g) point" of the 
    # reference tile based on the default llc tile layout.
    # For example, the tile to the "top" of tile 3 is tile 7 because tile 7
    # connects to the "top" of tile 3.  Tile 9 is the the "right" of tile 5
    # because tile 5 connects to the "right" of tile 5. Some tiles have are 
    # no adjacent neighbors to the right or top, or in the corner
    # corner: e.g., tiles 10 and 13 has no tile to the right 
    #
    # One use of this routine is to complete a GRID tile so that it contains
    # all of the parameters that describe the location and distances associated
    # with the (ni, nj) tracer grid cells associated with the same tile.  
    # These include the (ni+1, nj+1) XG and YG coordinates that define the 
    # corners of its tracer grid cells.  . 
    #
    # In the ECCO v4 netcdf tile files indexing of variables begins with the 
    # leftmost and bottommost points.  For variables on:
    # 'u' points : X[0,0] is at the bottom leftmost 'u' point to the left of 
    #              the center of the bottom leftmost tracer cell.
    # 'v' points : V[0,0] is at the bottom leftmost 'v' point below 
    #              the center of the bottom leftmost tracer cell.   
    # 'g' points : G[0,0] is at the bottom leftmost 'g' point at the bottom
    #              left corner of the bottom leftmost tracer cell
    # 'c' points : C[0,0] is in the middle of the bottom leftmost tracer cell
    #
    # Note, because tiles 8-13 of the llc grid are rotated 90 degrees 
    # relative to tiles 1-7 and because tile 7 is the Arctic cap, the meaning
    # of 'u' and 'v' is not 'east-west' and 'north-south' but 'left-right'
    # and 'top-bottom' relative to the tracer cell center.  After a 90 degree
    # rotation of tiles 8-13, the 'u' and 'v' points for tiles 1-6 and 8-13 
    # are approximately aligned to the east/west and north/south relative 
    # to the tracer cell 'c' points.  However, tile 7 must always be considered
    # a special case.
    #
    # variable arguments include
    # "right" -> the unrotated tile along the "right" edge.
    #          FOR 'G' AND 'U' POINTS
    #          ex.  for tile 2 the tile along the "right" edge is in tile 5
    #          ex.  for tile 9 the tile along the "right" edge is in tile 10
    #          ex.  for tile 7 the tile along the "right" edge is in tile 8
    #          
    # "top" -> the unrotated tile along the "top" edge.
    #          FOR 'G' AND 'V' POINTS
    #          ex.  for tile 2 the tile along the "top" edge is in tile 3
    #          ex.  for tile 9 the tile along the "top" edge is in tile 12
    #          ex.  for tile 7 the tile along the "top" edge is in tile 11 
    #               (yes, tile 11 is rotated we take care of that)
    #
    # "corner" -> the unrotated tile on the "corner (g) spot"  
    #          ONLY FOR SOME 'G' POINTS LIKE XG, YG
    #          ex.  for tile 2 the tile with the "corner" is in tile 6
    #          ex.  for tile 9 the tile with the "corner" is in tile 13
    #          ex.  for tile 7 the tile with the "corner" is nowhere
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # defaults
    add_right  = False
    add_top    = False
    add_corner = False
    
    right_arr = []
    top_arr = []
    corner_arr = []
    new_arr = []
    
    #%%
    for key in kwargs:
        if key == "right":
            add_right = True
            right_arr = kwargs[key]
        elif key == "top":
            add_top = True
            top_arr = kwargs[key]
        elif key == "corner" :
            add_corner = True
            corner_arr = kwargs[key]
    
    if add_right == False and add_top == False and add_corner == False:
        print "You need to indicate at least one side to append, returning an empty array"

    else:
            
        ni = llcN
        nj = llcN
        
        new_arr = np.copy(orig_arr)
        
        # expand the size of the array so that it can hold the new border values
        # fill border values with nans.  padding goes like (before, after)
        #
        # ex 1: two dimensions, pad one column along the end of the j dimension 
        #          pad_width=((0,0), (0,1))
        # ex 2: three dimensions, pad one column along the end of the i dimension 
        #          pad_width=((0,0), (0,1), (0,0))
        # ex 3: three dimensions, pad one column at the start of the i dimension
        #       and one at the end of the j dimension
        #          pad_width=((0,0), (1,0), (0,1))
        # 
        num_dims = new_arr.ndim

        if num_dims == 2:
            new_arr=np.pad(new_arr, pad_width=((0,pad_i), (0,pad_j)), 
                         mode='constant', constant_values = np.nan)

        elif num_dims == 3:
            new_arr=np.pad(new_arr, pad_width=((0,0),(0,pad_i), (0,pad_j)), 
                         mode='constant', constant_values = np.nan)

        elif num_dims == 4:
            new_arr=np.pad(new_arr, pad_width=((0,0),(0,0),(0,pad_i), (0,pad_j)), 
                         mode='constant', constant_values = np.nan)

        else:
            print 'appending rows or columns in this routine requires 2,3, or 4D arrays'
            
        
        if (tile_index == 1 or tile_index ==  2 or tile_index == 8 or 
            tile_index == 9 or tile_index == 10):
            
            if add_right:
                new_arr[...,0:nj,-1] = right_arr[...,:,0]

            if add_top :
                new_arr[...,-1,0:ni] = top_arr[..., 0,:]

            if add_corner :
                new_arr[...,-1,-1]   = corner_arr[..., 0,0]
         
        elif tile_index == 3:
            if add_right:
                new_arr[...,0:nj,-1] = right_arr[..., :,0]

            if add_top:
                # tricky because tile 7 is rotated relative to 3
                if point_type == 'g':            
                    new_arr[...,-1,1:ni+1] = top_arr[..., ::-1,0]                    
                elif point_type == 'v':
                    new_arr[...,-1,:] = top_arr[..., :,0]
                    
        elif (tile_index == 4 or tile_index == 5):
            if add_right:
                if point_type == 'g':
                    new_arr[...,1:nj+1,-1] = right_arr[..., 0,::-1]
                    
                elif point_type == 'u':
                    new_arr[...,-1] = right_arr[..., 0,::-1]
                    
            if add_top:
                new_arr[...,-1,0:ni] = top_arr[...,0,:]
                    
        elif tile_index == 6:
            if add_right:
                if point_type == 'g':
                    new_arr[...,1:nj+1,-1] = right_arr[...,0,::-1]
                    
                elif point_type == 'u':
                    new_arr[...,-1] = right_arr[...,0,::-1]
                    
            if add_top:
                new_arr[...,-1,0:ni] = top_arr[...,0,:]
                    
        elif tile_index == 7:
            if add_right:
                new_arr[...,0:nj,-1] = right_arr[...,:,0]

            if add_top:
                if point_type == 'g':
                    # tricky because tile 11 is rotated relative to 7
                    new_arr[...,-1,1:ni+1] = top_arr[...,::-1,0]

                elif point_type == 'v':
                    new_arr[...,-1,:] = top_arr[...,::-1,0]
        
        elif (tile_index == 11 or tile_index == 12 or tile_index == 13):
            if add_right:
                new_arr[...,0:ni,-1] = right_arr[...,:,0]
                
            if add_top:
                if point_type == 'g':
                    new_arr[...,-1,1:nj+1] = top_arr[...,::-1,0]

                elif point_type == 'v':
                    new_arr[...,-1,:] = top_arr[...,::-1,0]

            if add_corner : 
                # for tiles 12 and 13 the 'corner g point' is
                # in the top left point, unlike tiles 1-6 for which the 'corner
                # g point is the top right point.  tile 11 does not have a 
                # top left point because of the orientation of tile 7.
                new_arr[...,-1,0] = corner_arr[...,0,0]

    #%%
        
    return new_arr
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def add_borders_to_GRID_tiles(gds):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # A workhorse routine that applies the missing 'u', 'v', and 'g' point grid
    # parameters along the 'top' and 'eastern' edges of the 13 tiles.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # extract current dimensions of the grid tiles.
    ni = len(gds.j)  
    nj = len(gds.i)  
    nk = len(gds.k)  
    llcN = ni
    
    # define new coordiantes.
    i_g = np.arange(1, nj+2)
    j_g = np.arange(1, ni+2)
    i = np.arange(1, nj+1)
    j = np.arange(1, ni+1)
    k = np.arange(1, nk+1)
    
            
    #%%
    # FIRST, MAKE THE PARAMETER DATASET FOR GRID PARAMETERS SITUATED ON THE 
    # CORNER 'G' POINTS.  THERE ARE ONLY THREE: XG, YG and RAZ
    print "\n>>> ADDING BORDERS TO GRID TILES\n"
    
    print "G points"     
    vars = ['XG','YG','RAZ']
    # the new arrays will be one longer in the i and j directions, we will
    # pad on row to the top and one column to the right.
    pad_i = 1; pad_j = 1

    GRID_G = []
    for tile_index in range(1,14):  
        if tile_index in gds.tile.values:
                
            for idx, var in enumerate(vars):
                # get the indexes for the adjacent tiles            
                right_tile_index, top_tile_index, corner_tile_index = \
                    get_llc_tile_border_mapping(tile_index)
    
                # pull those tiles out if there are neighbors (index > 0)
                if right_tile_index > 0:
                    right_arr = gds[var].sel(tile=right_tile_index)
                if top_tile_index > 0:
                    top_arr = gds[var].sel(tile=top_tile_index)
                if corner_tile_index > 0:
                    corner_arr = gds[var].sel(tile=corner_tile_index)
                
                # save the original array for this variable.
                orig_arr = deepcopy(gds[var].sel(tile=tile_index))    
    
                # for the 'G' points there is always an adjacent top tile
                # but not always an adjacent right or corner tile.  these are 
                # the different possibilities: 
                # (1) right, top, and corner; (2) right, top            
                # (3) corner, top; (4) top only 
                if right_tile_index > 0 and corner_tile_index > 0: 
                    #right, top and corner
                    new_arr=append_border_to_tile(orig_arr, tile_index, 
                                                  'g',pad_i, pad_j, llcN,
                                                  right = right_arr, 
                                                  top = top_arr, 
                                                  corner = corner_arr)
                elif right_tile_index > 0: # right, top
                    new_arr=append_border_to_tile(orig_arr, tile_index,
                                                  'g',pad_i, pad_j, llcN,
                                                  right = right_arr, 
                                                  top = top_arr)
                elif corner_tile_index > 0: # corner and top
                    new_arr=append_border_to_tile(orig_arr, tile_index,
                                                  'g',pad_i, pad_j, llcN,
                                                  top = top_arr,
                                                  corner = corner_arr)
                else: # top only
                    new_arr=append_border_to_tile(orig_arr, tile_index, 
                                                  'g',pad_i, pad_j, llcN,
                                                  top=top_arr)
                # make a new dataset and define the ecoordinates.
                tmp_DS = xr.Dataset({var: (['j_g','i_g'], new_arr)},
                                     coords={'i_g': i_g, 'j_g':j_g})
                
                # copy the attributes of the original grid dataste
                tmp_DS[var].attrs = gds[var].attrs
                
                # add the new coordinate, tile.
                tmp_DS.coords['tile'] = tile_index
                
                # if this is the first variable then create the Dataset
                if idx == 0:
                    grid_g_tile = tmp_DS
                else:
                    # otherwise, merge the new Dataset with the existing Dataset
                    # for this tile.
                    grid_g_tile = xr.merge([grid_g_tile, tmp_DS])
                    
    
            # if this is the first tile, define GRID_G to be the Dataset
            if tile_index == 1:
                GRID_G = grid_g_tile
            else:
                # otherwise, concatenate the new grid dataset corresponding with
                # this tile to the existing Dataset along the 'tile' dimension
                GRID_G = xr.concat([GRID_G, grid_g_tile], 'tile')

        # end of tile loop
    
    #%%
    # NEXT MAKE THE PARAMETER DATASET FOR GRID PARAMETERS SITUATED ON THE 
    # U POINTS.  THERE ARE FOUR: DXC, DYG, and HFACW, land_u
    
    print "U points"    
    vars = ['DXC','DYG', 'hFacW', 'land_u']
    
    # because tiles 8-13 are rotated relative to tiles 1-6, the names of 
    # of variables are different between them.  When tile 4 needs DXC from
    # tile 10, we need to look for DYC in tile 10, etc.
    rot_vars = ['DYC', 'DXG', 'hFacS', 'land_v']
    
    # tiles for which the adjacent tile to the right is rotated relative to itself
    rot_tiles = {4, 5, 6}

    # the new arrays will be one longer in the j direction, we will
    # pad one column to the right.
    pad_i = 0 # we do not pad in first dimension (y)
    pad_j = 1 # add one to the second dimension (x)
    
    for tile_index in range(1,14):  

        # Proceed if this tile is in the Dataset
        if tile_index in gds.tile.values:
            
            # we only care about the "right_tile_index" here
            right_tile_index, top_tile_index, corner_tile_index = \
                get_llc_tile_border_mapping(tile_index)
            
            # loop over all vars 
            for idx, var in enumerate(vars):
                #print tile_index, idx, var
                
                # one dimension is tile
                num_dims = gds[var].ndim - 1

                orig_arr = gds[var].sel(tile=tile_index)    
    
                # check to see that there is a neighbor to the right
                # tiles 10 and 13 don't have one.)
                if right_tile_index > 0:
                    # if we have a tile whose neighbor to the right is 
                    # rotated relative to itself, i.e., tile 6's neighbor to the
                    # right is tile 8, then we need to pull the 'rot_var' out 
                    # of the neighbor instead of 'var'
                    if tile_index in rot_tiles:
                        right_arr = \
                            gds[rot_vars[idx]].sel(tile=right_tile_index)
                    else:
                        right_arr = gds[var].sel(tile=right_tile_index)
            
                    new_arr=append_border_to_tile(orig_arr, tile_index,
                                                  'u',pad_i, pad_j, llcN,
                                                  right = right_arr)            
                else:
                    # if there is no neighbor to the right
                    # pad the orig_array with nans on its right hand side
                    # applies to tiles 10 and 13, they have no tile to 
                    # their right (south pole.)
                    if num_dims == 2:
                        new_arr = np.pad(orig_arr, 
                                         pad_width=((0,pad_i), (0,pad_j)), 
                                         mode='constant', 
                                         constant_values = np.nan)
                    elif num_dims == 3:
                        new_arr = np.pad(orig_arr, 
                                         pad_width=((0,0), (0,pad_i), 
                                                    (0,pad_j)), 
                                         mode='constant', 
                                         constant_values = np.nan)
    
                # create a new Dataset with the variable and give it new 
                # dimensions.
                if num_dims == 2:
                    tmp_DS = xr.Dataset({var: (['j','i_g'], new_arr)},
                                         coords={'i_g': i_g, 'j':j})
                elif num_dims == 3:
                    tmp_DS = xr.Dataset({var: (['k','j','i_g'], new_arr)},
                         coords={'k':k, 'j': j, 'i_g':i_g})
                    
                #-- finished handling 2D or 3D case
                tmp_DS[var].attrs = gds[var].attrs
                tmp_DS.coords['tile'] = tile_index
            
                if idx == 0:
                    grid_u_tile = tmp_DS
                else:
                    grid_u_tile = xr.merge([grid_u_tile, tmp_DS])
    
                tmp_DS = []
                # end of variable loop
                
            if tile_index == 1:
                GRID_U = grid_u_tile
            else:
                GRID_U = xr.concat([GRID_U, grid_u_tile], 'tile')
    
            grid_u_tile = []
    
    
    #%%
    # FINALLY MAKE THE PARAMETER DATASET FOR GRID PARAMETERS SITUATED ON THE 
    # V POINTS.  THERE ARE FOUR: DYC, DXG, hFacS, and land_v
    
    print "V points"
    vars = ['DYC','DXG', 'hFacS', 'land_v']
    
    # tiles for which the adjacent tile to the top is rotated relative to itself
    rot_tiles = {3, 7, 11, 12, 13}
    # and the names of the variables on rot_tiles that will be used 
    # remember, x-->y, y-->x w-->s after rotation.
    rot_vars = ['DXC', 'DYG','hFacW', 'land_u']
    
    pad_i = 1 #  add one to the first dimension (y)
    pad_j = 0 #  do not pad in second dimension (x)
    
    for tile_index in range(1,14):

        # Proceed if this tile is in the Dataset
        if tile_index in gds.tile.values:
            grid_u_tile = []
            
            # we only care about "top_tile_index" since are are adding to the
            # top only.
            right_tile_index, top_tile_index, corner_tile_index = \
                get_llc_tile_border_mapping(tile_index)
            
            for idx, var in enumerate(vars):
    
                num_dims = gds[var].ndim - 1                

                orig_arr = gds[var].sel(tile=tile_index)    
    
                if tile_index in rot_tiles:
                    top_arr = gds[rot_vars[idx]].sel(tile=top_tile_index)
                else:
                    top_arr = gds[var].sel(tile=top_tile_index)
        
                new_arr=append_border_to_tile(orig_arr, tile_index,
                                              'v',pad_i, pad_j, llcN,
                                              top = top_arr)
                if num_dims == 2:
                    tmp_DS = xr.Dataset({var: (['j_g','i'], new_arr)},
                                         coords={'j_g': j_g, 'i':i})
                elif num_dims == 3:
                    # we have three dimensions with the third dimension being 
                    # depth (hFacS)
                    tmp_DS = xr.Dataset({var: (['k','j_g','i'], new_arr)},
                                         coords={'k':k, 'j_g': j_g, 'i':i})
                    
                # Give the dataset the same attributes
                tmp_DS[var].attrs = gds[var].attrs
                # and add the tile attribute
                tmp_DS.coords['tile'] = tile_index
            
                if idx == 0:
                    grid_v_tile = tmp_DS
                else:
                    grid_v_tile = xr.merge([grid_v_tile, tmp_DS])
    
                tmp_DS = []
    
            # concatenate the new Dataset for this tile with GRID_V (or define it
            # if tile == 1)            
            if tile_index == 1:
                GRID_V = grid_v_tile
            else:
                GRID_V = xr.concat([GRID_V, grid_v_tile], 'tile')
    
            grid_v_tile = []


    # Now merge the rotated grid parameters.
    GRID_Z = gds[['RB', 'RC','RF','DRC','DRF']]
    GRID = gds.drop(['DYC','DXG', 'hFacS','DXC', 'DYG','hFacW','XG','YG',
                     'RAZ','RB','RC','RF','DRC','DRF'])
    GRID = xr.merge([GRID, GRID_G, GRID_U, GRID_V, GRID_Z])
    GRID.attrs = gds.attrs
    #%%
    return GRID
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
def get_llc_tile_border_mapping(tile_index):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # this routine provides the tile number for those tiles to the 'right'
    # 'top' and 'corner' of the reference tile (tile_index) in the
    # original llc tile layout if it exists
    # In several cases there is no adajacent tile (e.g., tile 13 has 
    # no tile to its right because it ends at the south pole.

    # below is the mapping between the tiles for the 'right','top', and 'top
    # right corner'
    
    if tile_index == 1:
        right_tile_index = 4
        top_tile_index = 2
        corner_tile_index= 5
    elif tile_index == 2:
        right_tile_index = 5
        top_tile_index = 3
        corner_tile_index = 6
    elif tile_index == 3:
        right_tile_index = 6
        top_tile_index = 7
        # does not exist 
        corner_tile_index = -1
    elif tile_index == 4:
        right_tile_index = 10
        top_tile_index = 5
        # does not exist 
        corner_tile_index = -1
    elif tile_index == 5:
        right_tile_index = 9
        top_tile_index = 6  
        corner_tile_index = -1
    elif tile_index == 6:
        right_tile_index = 8
        top_tile_index = 7
        corner_tile_index = -1
    elif tile_index == 7:
        right_tile_index = 8
        top_tile_index = 11
        corner_tile_index = -1
    elif tile_index == 8:
        right_tile_index = 9
        top_tile_index = 11
        corner_tile_index = 12
    elif tile_index == 9:
        right_tile_index = 10
        top_tile_index = 12
        corner_tile_index = 13
    elif tile_index == 10:
        right_tile_index = -1
        top_tile_index = 13
        corner_tile_index = -1
    elif tile_index == 11:
        right_tile_index = 12
        top_tile_index = 3
        corner_tile_index = -1 # no corner defined because of tile 7's rotation
    elif tile_index == 12:
        right_tile_index = 13
        top_tile_index = 2
        corner_tile_index = 3 # yes, corner is "top left"
    elif tile_index == 13:
        right_tile_index = -1
        top_tile_index = 1
        corner_tile_index = 2 # yes, corner is "top left"
    
    return right_tile_index, top_tile_index, corner_tile_index   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
            
