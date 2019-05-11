#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""ECCO v4 Python: tile_exchange.py

This module includes utility routines adding halos to individual llc tiles arrays. 

.. _ecco_v4_py Documentation :
   https://github.com/ECCO-GROUP/ECCOv4-py
"""
from __future__ import division,print_function
import numpy as np
import xarray as xr
from copy import deepcopy




def append_border_to_tile(ref_arr, tile_index, point_type, llcN, **kwargs):
    """

    This routine takes an array corresponding to an llc tile (the reference
    array) and appends one of three possible partial "halos":
    1) a new row along the "top" of the array (new 'v' points) 
    2) a new column to the "right" of the array (new 'u' points)
    3) a new single value in a "corner (g) point" of the array (new 'g' point).
    
    After the routine the new array will have dimension depending on where
    the variable is on the Arakawa-C grid
    1) (nx, ny+1) for appending values from variables on 'v' points 
    2) (nx+1, ny) for appending values from variables on 'u' points  
    3) (nx+1, ny+1) for appending values from variables on 'g' points
    
    Appending new values from adjacent neighbors is only permitted for tiles 
    that connect to the "top", "right", and/or "corner (g) point" of the 
    reference tile based on the default llc tile layout. 

    For example, the tile to the "top" of tile index 2 is tile 6 because tile 6 
    connects to the "top" of tile 2.  Tile 8 is the the "right" of tile 4
    because tile 4 connects to the "right" of tile 4. Some tiles have are no 
    adjacent neighbors to the right or top, or in the corner: e.g., tiles 9 
    and 12 have no tile to their right
    
    One use of this routine is to complete a `grid` tile so that it 
    contains all of the parameters that describe the location and distances 
    associated  with the (ni, nj) tracer grid cells associated with the same 
    tile. These include the (ni+1, nj+1) XG and YG coordinates that define the
    corners of its tracer grid cells.
    
    In the ECCO v4 netcdf tile files indexing of variables begins with the 
    leftmost and bottommost points.  For variables on:
        
    'u' points : X[0,0] is at the bottom leftmost 'u' point to the 
    left of the center of the bottom leftmost tracer cell.
   
    'v' points : V[0,0] is at the bottom leftmost 'v' point below
    the center of the bottom leftmost tracer cell.
  
    'g' points : G[0,0] is at the bottom leftmost 'g' point at the  
    bottom left corner of the bottom leftmost tracer cell 
    
    'c' points : C[0,0] is in the middle of the bottom leftmost tracer cell
    
    Note
    ---- 
        The values in the "halo" are taken from adjacent tiles that are 
        passed in kwargs.  The orientation of the reference tile and the 
        adjacent tiles must be in the original 13-tile llc layout (i.e., this
        routine must be called prior to any rotation or reorientation).

    Note
    ----  
        In the MITgcm the meaning of 'u' and 'v' is not east-west (zonal) 
        and north-south (meridional) in a geographical sense.  Instead 'u'
        and 'v' are the direction of flow in the 'x' and 'y' directions for 
        the local orientation a tracer grid cell.  
        
        For the llc grid, the tracer cells in tile index 0-5 are oriented 
        such that their 'x' and 'y'
        directions are approximately east-west and north-south, respectively.  
        However, tiles 7-12 are rotated by 90 degrees relatively to tiles 0-6
        The 'x' direction in tiles 7-12 is the negative 'y' direction of 
        tiles 0-5 and the 'y' directin in tiles 7-12 is the positive 'x' 
        direction in tiles 0-5  
        
        The polar cap tile 6 is a special case. There is no rotation of 
        tile 6 that will align its 'x' and 'y' grid 
        to be zonal or meridional in a geographical sense.
    
    
    Parameters
    ----------
    ref_arr : ndarray
        A string with the directory of the binary file to open
    tile_index : int
        the index of the reference tile [1-13].
    point_type : string
        The type of partial halo to put on the reference tile, one of 'u','v',or 'g'.  
        'u' add one column to the right of the array
        'v' add one row to the top of the array
        'g' add one column to the right of the array and one row to the top of the array
    llcN : int
        the size of the llc grid.  For ECCO v4, we use the llc90 domain so `llcN` would be 90
    **kwargs
        right : ndarray
            the unrotated tile along the "right" edge. for 'g' and 'u' points
            for tile 0, tile 3
            for tile 7, tile 8
            for tile 5, tile 7
        top : ndarray
            the unrotated tile along the "top" edge. for 'g' and 'v' points
            for tile 0, tile 1
            for tile 7, tile 10
            for tile 5, tile 6 (rotated)
        corner : ndarray
            the unrotated tile in the "top right corner" for 'g' points
            for tile 0, tile 4
            for tile 7, tile 11
            for tile 5, tile 7

    Returns
    -------
    ndarray
        a new array that has the halo points around it.

    Examples
    --------
    Add a halo on `ref_array` to the top row and right column at the 'g' points.

    ex 1. This tile has adjacent tiles to its right top and top right corner.

    >>> newarr = append_border_to_tile(ref_arr, tile_index, 
                                      'g', llcN,
                                      right = right_arr, 
                                      top = top_arr, 
                                      corner = corner_arr)

    ex 2. Add a halo on `ref_array` to the right column at the 'u' points.
    
    >>> newarr = append_border_to_tile(ref_arr, tile_index,
                                       'u', llcN, right = right_arr)
    """

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
            if isinstance(kwargs[key], xr.core.dataarray.DataArray):
                add_right = True
                right_arr = kwargs[key]
        elif key == "top":
            if isinstance(kwargs[key], xr.core.dataarray.DataArray):
                add_top = True
                top_arr = kwargs[key]
        elif key == "corner" :
            if isinstance(kwargs[key], xr.core.dataarray.DataArray):
                add_corner = True
                corner_arr = kwargs[key]

    if add_right == False and add_top == False and add_corner == False:
        raise ValueError('You need to have at least one side to append, \
                         returning an empty array')

    ni = llcN
    nj = llcN
    
    print ('ar, at, ac ', add_right, add_top, add_corner)
    #print ni, nj, llcN
    
    new_arr = np.copy(ref_arr)
    """
    expand the size of the array so that it can hold the new border values
    fill border values with nans.  padding goes like (before, after)
    
    ex 1: two dimensions, pad one column along the end of the j dimension 
             pad_width=((0,0), (0,1))
    ex 2: three dimensions, pad one column along the end of the i dimension 
             pad_width=((0,0), (0,1), (0,0))
    ex 3: three dimensions, pad one column at the start of the i dimension
          and one at the end of the j dimension
             pad_width=((0,0), (1,0), (0,1))
    """

    if point_type == 'g':            
        pad_i = 1 # add one row (y)
        pad_j = 1 # add one column (x)
    elif point_type == 'v':
        pad_i = 1 # add one row (y)
        pad_j = 0 # do not add one column (x)
    elif point_type == 'u':
        pad_i = 0 # do not add one row (x)
        pad_j = 1 # add one column (x)
        
    # print 'num dims ', num_dims


    #if num_dims == 2:
    new_arr=np.pad(new_arr, 
        pad_width=((0,pad_i), (0,pad_j)), 
        mode='constant', constant_values = np.nan)

    #elif num_dims == 3:
    #    new_arr=np.pad(new_arr, 
    #        pad_width=((0,0),(0,pad_i), (0,pad_j)),
    #        mode='constant', constant_values = np.nan)

    #elif num_dims == 4:
    #    new_arr = np.pad(new_arr, 
    #        pad_width=((0,0),(0,0),(0,pad_i), (0,pad_j)), 
    #        mode='constant', constant_values = np.nan)

    #else:
    #    raise ValueError('appending rows or columns in this routine requires \
    #                     2,3, or 4D arrays')
    
    
    # add right
    if add_right:
        if tile_index in (0,1,2,6,7,8,10,11):
            new_arr[0:nj,-1] = right_arr[0:nj,0] ## verified
        elif tile_index in (3,4,5):
            tmp = right_arr[0, :nj]
            
            if point_type == 'g':
                new_arr[1:nj+1,-1] = tmp[::-1]  ## verified
            elif point_type == 'u':
                new_arr[0:nj,-1] = tmp[::-1]  ## not yet verified
                
        elif tile_index in (9,12): # nothing to the right
            new_arr[:,-1] = np.nan
        else:
            print ('error, tile_index add_right')
            
    if add_top:
        if tile_index in (0,1,3,4,5,7,8,9):
            new_arr[-1,0:ni] = top_arr[0,0:ni] ## verified
        elif tile_index in (2, 6, 10,11,12):
            tmp = top_arr[:,0]
            if point_type == 'g':
                new_arr[-1,1:ni+1] = tmp[::-1] ## verified
            elif point_type == 'v':
                new_arr[-1,0:ni] = tmp[::-1]  ## not yet verified
        else:
            print ('error, tile index add_top')
  
    if add_corner:
        if tile_index in (0,1,2,6,7,8,10):
            new_arr[-1,-1] = corner_arr[0,0]
            print ('tile_index, corner_arr ', tile_index, corner_arr[0,0])
        elif tile_index in (11,12):
            new_arr[-1,0] = corner_arr[0,0] # top left corner is missing.
        elif tile_index == 3:
            new_arr[-1,-1] = corner_arr[0,0] # no bottom right corner
        elif tile_index in (4,5): # bottom right corner
            new_arr[0,-1] = corner_arr[0,0]
        elif tile_index in (9):  # no top right corner.  no bottom right corner
            new_arr[-1,-1] = np.nan
        else:
            print ('error, tile index add corner')
#    if (tile_index == 1 or tile_index ==  2 or tile_index == 8 or 
#        tile_index == 9 or tile_index == 10):
#        
#        if add_right:
#            new_arr[...,0:nj,-1] = right_arr[...,:,0]
#
#        if add_top :
#            new_arr[...,-1,0:ni] = top_arr[..., 0,:]
#
#        if add_corner :
#            new_arr[...,-1,-1]   = corner_arr[..., 0,0]
#     
#    elif tile_index == 3:
#        if add_right:
#            new_arr[...,0:nj,-1] = right_arr[..., :,0]
#
#        if add_top:
#            # tricky because tile 7 is rotated relative to 3
#            if point_type == 'g':            
#                new_arr[...,-1,1:ni+1] = top_arr[..., ::-1,0]                    
#            elif point_type == 'v':
#                new_arr[...,-1,:] = top_arr[..., :,0]
#                
#    elif (tile_index == 4 or tile_index == 5):
#        if add_right:
#            if point_type == 'g':
#                new_arr[...,1:nj+1,-1] = right_arr[..., 0,::-1]
#                
#            elif point_type == 'u':
#                new_arr[...,-1] = right_arr[..., 0,::-1]
#                
#        if add_top:
#            new_arr[...,-1,0:ni] = top_arr[...,0,:]
#                
#    elif tile_index == 6:
#        if add_right:
#            if point_type == 'g':
#                new_arr[...,1:nj+1,-1] = right_arr[...,0,::-1]
#                
#            elif point_type == 'u':
#                new_arr[...,-1] = right_arr[...,0,::-1]
#                
#        if add_top:
#            new_arr[...,-1,0:ni] = top_arr[...,0,:]
#                
#    elif tile_index == 7:
#        if add_right:
#            new_arr[...,0:nj,-1] = right_arr[...,:,0]
#
#        if add_top:
#            if point_type == 'g':
#                # tricky because tile 11 is rotated relative to 7
#                new_arr[...,-1,1:ni+1] = top_arr[...,::-1,0]
#
#            elif point_type == 'v':
#                new_arr[...,-1,:] = top_arr[...,::-1,0]
#    
#    elif (tile_index == 11 or tile_index == 12 or 
#          tile_index == 13):
#        if add_right:
#            new_arr[...,0:ni,-1] = right_arr[...,:,0]
#            
#        if add_top:
#            if point_type == 'g':
#                new_arr[...,-1,1:nj+1] = top_arr[...,::-1,0]
#
#            elif point_type == 'v':
#                new_arr[...,-1,:] = top_arr[...,::-1,0]
#
#        if add_corner :
#            """             
#            for tiles 12 and 13 the 'corner g point' is
#            in the top left point, unlike tiles 1-6 for which the 
#            'corner' g point is the top right point.  tile 11 does not
#            have a top left point because of the orientation of tile 7.
#            """                
#            new_arr[...,-1,0] = corner_arr[...,0,0]
#        
    #%%
    return new_arr
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





def add_borders_to_DataArray_V_points(da_u, da_v):
    """
    A routine that adds a column to the "top" of the 'v' point 
    DataArray da_v so that every tracer point in the tile 
    will have a 'v' point to the "north" and "south"

    After appending the border the length of da_v in y will
    be +1 (one new row)
    
    This routine is pretty general.  Any tiles can be in the da_u and 
    da_v DataArrays but if the tiles to the "top" of the da_v tiles
    are not available then the new rows will be filled with nans.

    Parameters
    ----------
    da_u : DataArray
        The `DataArray` object that has tiles of a u-point variable
        Tiles of the must be in their original llc layout.

    da_v : DataArray
        The `DataArray` object that has tiles of the v-point variable that 
        corresponds with da_u.   (e.g., VVEL corresponds with UVEL)
        Tiles of the must be in their original llc layout.
       
    Returns
    -------
    da_v_new: DataArray
        a new `DataArray` object that has the appended values of 'v' along
        its top edge.  The lon_u and lat_u coordinates are lost but all
        other coordinates remain.
    """
    
    #%%
    # the j_g dimension will be incremented by one.
    j_g = np.arange(1, len(da_v.j_g)+2)
    
    # the i dimension is unchanged.
    i = da_v['i'].values
    llcN = len(i)
    
    # the k dimension, if it exists, is unchanged.
    if 'k' in da_v.dims:
        nk = len(da_v.k)  
        k = da_v['k'].values
    else:
        nk = 0

    # the time dimension, if it exists, is unchanged
    if 'time' in da_v.dims:
        time = da_v['time'].values
            
    #%%
    #print "\n>>> ADDING BORDERS TO V POINT TILES\n"
    
    # tiles whose tile to the top are rotated 90 degrees counter clockwise
    rot_tiles = {3, 7, 11, 12, 13}

    # the new arrays will be one longer in the j direction, +1 column
    pad_i = 1 # pad by one in y (one new row)
    pad_j = 0 # do not pad in x 

    # set the number of processed tiles counter to zero    
    num_proc_tiles = 0

    # find the number of non-tile dimensions
    if 'tile' in da_v.dims:
        num_dims = da_v.ndim - 1
    else:
        num_dims = da_v.ndim

    # loop through all tiles in da_v
    for tile_index in da_v.tile.values:
        
        # find out which tile is to the top of this da_v tile
        right_tile_index, top_tile_index, corner_tile_index = \
            get_llc_tile_border_mapping(tile_index)

        # if 'tile' exists as a dimension, select and copy this da_v tile
        if 'tile' in da_v.dims:
            ref_arr = deepcopy(da_v.sel(tile=tile_index))
        else:
            # otherwise we have a single da_v tile so make a copy of it
            ref_arr = deepcopy(da_v)

        # the append_border flag will be true if we have a tile to the top.
        append_border = False
        
        #print '\ncurrent da_v tile ', tile_index
        #print 'top tile index ', top_tile_index
        
        # check to see if there even is a tile to the top of da_v tile_index
        if top_tile_index > 0:
            #print 'there is a tile to the top of da_v tile ', tile_index

            # determine whether the tile to the top is rotated relative 
            # to da_v tile_index. if so we'll need da_u!
            if tile_index in rot_tiles:
                #print 'append with da_u tile ', top_tile_index

                if top_tile_index in da_u.tile.values:
                    #print 'we have da_u tile ', top_tile_index                      
                    
                    # see if we have multiple da_u tiles. 
                    if len(da_u.tile) > 1:
                        # pull out the one we need.
                        top_arr = da_u.sel(tile=top_tile_index)
                        append_border = True
                        #print 'appending from da_u tile ', top_tile_index
                    # there is only one da_u tile 
                    elif da_u.tile == top_tile_index:
                        # it is the one we need.
                        top_arr = da_u
                        append_border = True                        
                        #print 'appending from da_u tile ', top_tile_index                        
                    # something may have gone wrong.
                    else:
                        print('something is wrong with the da_u tile')
                        
                 # if we do not have the da_u tile, then we can't append!
                else:
                   print('we do not have da_u tile ', top_tile_index)

            # the values to append to the top come from another da_v tile
            else:
                #print 'append with da_v tile ', top_tile_index

                # see if we have the required da_v tile
                if top_tile_index in da_v.tile.values:
                    #print 'we have da_v tile ', top_tile_index                      

                    # see if we have multiple da_v tiles
                    if len(da_v.tile) > 1:
                        # pull out the one we need.
                        top_arr = da_v.sel(tile=top_tile_index)
                        append_border = True
                        #print 'appending from da_v tile ', top_tile_index
                    # if we only have one tile then something is wrong because
                    # the tile to the top of this da_v tile cannot be itself
                    #else:
                       # print 'tile to the top cannot be tile_index'
                # we do not have the required da_v tile.
                else:
                    print('we do not have da_v tile ', top_tile_index)

        # there is no tile to the top 
        else:
            print('there is no tile to the top of da_v tile ', tile_index)

        # if we have found a tile to the top we can do the appending
        if append_border:
            new_arr=append_border_to_tile(ref_arr, tile_index,
                                               'v', llcN,
                                               top = top_arr)            

        # if not then we will append an array of nans
        else:
            if num_dims == 2:
                pad = ((0, pad_i), (0, pad_j))
            elif num_dims == 3:
                pad = ((0, 0),     (0, pad_i), (0, pad_j))
            elif num_dims == 4:
                pad = ((0, 0),     (0, 0),     (0, pad_i), (0, pad_j))
            
            new_arr = np.pad(ref_arr, pad_width = pad, mode='constant',
                             constant_values = np.nan)

        # create a new DataArray
        if num_dims == 2:
            new_coords = [('j_g', j_g), ('i', i)]
        elif num_dims == 3 and nk > 0:
            new_coords = [('k', k), ('j_g', j_g), ('i', i)]
        elif num_dims == 3 and nk == 0:
            new_coords = [('time', time),('j_g', j_g), ('i', i)]
        elif num_dims == 4:
            new_coords = [('time', time), ('k', k), ('j_g', j_g), ('i', i)]
            
        tmp_DA = xr.DataArray(new_arr, name = da_v.name, coords=new_coords)

        # give the new DataArray the same attributes as da_v
        tmp_DA.attrs = da_v.attrs

        # give the new DataArray a tile coordinate
        tmp_DA.coords['tile'] = tile_index
    
        # increment the number of processed tiles counter by one
        num_proc_tiles += 1
        
        # set da_v_new equal to tmp_DA if this is the first processed tile
        if num_proc_tiles == 1:
            da_v_new = tmp_DA
        # otherwise, concatentate tmp_DA with da_v_new along the 'tile' dim
        else:
            da_v_new = xr.concat([da_v_new, tmp_DA],'tile')

        # reset tmp_DA
        tmp_DA = []

    # add all time coordinates to_da_v_new from da_v.
    for idx, var in enumerate(da_v.coords):
        if 'tim' in var:
            da_v_new[var] = da_v[var]

    da_v_new.attrs['padded'] = True            
    #%%
    return da_v_new
    #%%


def add_borders_to_DataArray_U_points(da_u, da_v):
    """
    A routine that adds a column to the "right" of the 'u' point 
    DataArray da_u so that every tracer point in the tile 
    will have a 'u' point to the "west" and "east"

    After appending the border the length of da_u in x 
    will be +1 (one new column)
    
    This routine is pretty general.  Any tiles can be in the da_u and 
    da_v DataArrays but if the tiles to the "right" of the da_u tiles
    are not available then the new rows will be filled with nans.
    
    Parameters
    ----------
    da_u : DataArray
        The `DataArray` object that has tiles of a u-point variable
        Tiles of the must be in their original llc layout.

    da_v : DataArray
        The `DataArray` object that has tiles of the v-point variable that 
        corresponds with da_u.   (e.g., VVEL corresponds with UVEL)
        Tiles of the must be in their original llc layout.
       
    Returns
    -------
    da_u_new: DataArray
        a new `DataArray` object that has the appended values of 'u' along
        its right edge.  The lon_u and lat_u coordinates are lost but all
        other coordinates remain.
    """
    
    #%%
    # the i_g dimension will be incremented by one.
    i_g = np.arange(1, len(da_u.i_g)+2)
    
    # the j dimension is unchanged.
    j = da_u['j'].values
    llcN = len(j)
    
    # the k dimension, if it exists, is unchanged.
    if 'k' in da_u.dims:
        nk = len(da_u.k)  
        k = da_u['k'].values
    else:
        nk = 0

    # the time dimension, if it exists, is unchanged
    if 'time' in da_u.dims:
        time = da_u['time'].values
            
    #%%
    #print "\n>>> ADDING BORDERS TO U POINT TILES\n"
    
    # tiles whose tile to the right are rotated 90 degrees counter clockwise
    # to add borders from tiles 4, 5, or 6 we need to use the da_v fields
    rot_tiles = {4, 5, 6}

    # the new arrays will be one longer in the j direction, +1 column
    pad_j = 1 # add one to the second dimension (x)
    pad_i = 0 # we do not pad in first dimension (y)    

    # set the number of processed tiles counter to zero    
    num_proc_tiles = 0

    # find the number of non-tile dimensions
    if 'tile' in da_u.dims:
        num_dims = da_u.ndim - 1
    else:
        num_dims = da_u.ndim
            
    # loop through all tiles in da_u
    for tile_index in da_u.tile.values:
        
        # find out which tile is to the right of this da_u tile
        right_tile_index, top_tile_index, corner_tile_index = \
            get_llc_tile_border_mapping(tile_index)
        
        # if 'tile' exists as a dimension, select and copy the proper da_u tile 
        if 'tile' in da_u.dims:
            ref_arr = deepcopy(da_u.sel(tile=tile_index))
        else:
            # otherwise we have a single da_u tile so make a copy of it
            ref_arr = deepcopy(da_u)

        # the append_border flag will be true if we have a tile to the right.
        append_border = False
        
        #print '\ncurrent da_u tile ', tile_index
        #print 'right tile index ', right_tile_index
        
        # check to see if there even is a tile to the right of da_u tile_index
        # tiles 10 and 13 don't have one!
        if right_tile_index > 0:
            #print 'there is a tile to the right of da_u tile ', tile_index

            # determine whether the tile to the right is rotated relative 
            # to da_u tile_index. if so we'll need da_v!
            if tile_index in rot_tiles:
                #print 'append with da_v tile ', right_tile_index

                if right_tile_index in da_v.tile.values:
                    #print 'we have da_v tile ', right_tile_index                      
                    
                    # see if we have multiple da_v tiles. 
                    if len(da_v.tile) > 1:
                        # pull out the one we need.
                        right_arr = da_v.sel(tile=right_tile_index)
                        append_border = True
                        #print 'appending from da_v tile ', right_tile_index
                    # there is only one da_v tile 
                    elif da_v.tile == right_tile_index:
                        # it is the one we need.
                        right_arr = da_v
                        append_border = True                        
                        #print 'appending from da_v tile ', right_tile_index                        
                    # something may have gone wrong.
                    else:
                        print('something is wrong with the da_v tile')
                        
                # if we do not have the da_v tile, then we can't append!
                else:
                    print('we do not have da_v tile ', right_tile_index)

            # the values to append to the top come from another da_u tile
            else:
                #print 'append with da_u tile ', right_tile_index

                # see if we have the required da_u tile
                if right_tile_index in da_u.tile.values:
                    #print 'we have da_u tile ', right_tile_index                      

                    # see if we have multiple da_u tiles
                    if len(da_u.tile) > 1:
                        # pull out the one we need.
                        right_arr = da_u.sel(tile=right_tile_index)
                        append_border = True
                        #print 'appending from da_u tile ', right_tile_index
                    # if we only have one tile then something is wrong because
                    # the tile to the right of this da_u tile cannot be itself
                    else:
                        print('tile to the right cannot be tile_index')
                # we do not have the required da_u tile.
                else:
                    print('we do not have da_u tile ', right_tile_index)

        # there is no tile to the right 
        #else:
        #    print 'there is no tile to the right of da_u tile ', tile_index

        # if we have found a tile to the right we can do the appending
        if append_border:
            new_arr=append_border_to_tile(ref_arr, tile_index,
                                               'u', llcN,
                                               right = right_arr)            

        # if not then we will append an array of nans
        else:
            if num_dims == 2:
                pad = ((0, pad_i), (0, pad_j))
            elif num_dims == 3:
                pad = ((0, 0),     (0, pad_i), (0, pad_j))
            elif num_dims == 4:
                pad = ((0, 0),     (0, 0),     (0, pad_i), (0, pad_j))
            
            new_arr = np.pad(ref_arr, pad_width = pad, mode='constant',
                             constant_values = np.nan)

        # create a new DataArray
        if num_dims == 2:
            new_coords = [('j', j), ('i_g', i_g)]
        elif num_dims == 3 and nk > 0:
            new_coords = [('k', k), ('j', j), ('i_g', i_g)]
        elif num_dims == 3 and nk == 0:
            new_coords = [('time', time),('j', j), ('i_g', i_g)]
        elif num_dims == 4:
            new_coords = [('time', time), ('k', k), ('j', j), ('i_g',i_g)]
            
        tmp_DA = xr.DataArray(new_arr, name = da_u.name, coords=new_coords)

        # give the new DataArray the same attributes as da_u
        tmp_DA.attrs = da_u.attrs

        # give the new DataArray a tile coordinate
        tmp_DA.coords['tile'] = tile_index
    
        # increment the number of processed tiles counter by one
        num_proc_tiles += 1
        
        # set da_u_new equal to tmp_DA if this is the first processed tile
        if num_proc_tiles == 1:
            da_u_new = tmp_DA
        # otherwise, concatentate tmp_DA with da_u_new along the 'tile' dim
        else:
            da_u_new = xr.concat([da_u_new, tmp_DA],'tile')

        # reset tmp_DA
        tmp_DA = []

    # add all time coordinates to_da_u_new from da_u.
    for idx, var in enumerate(da_u.coords):
        if 'tim' in var:
            da_u_new[var] = da_u[var]

    da_u_new.attrs['padded'] = True            
    #%%       
    return da_u_new
    #%%
    
    
    
def add_borders_to_DataArray_G_points(da_g):
    """
    A routine that adds a row and column to the "top" and "right"
    of the 'g' point DataArray da_g so that every tracer point in the tile 
    will have a 'g' point on all of its corners

    After appending the border the length of da_g will be +1 in both 
    i_g and j_g
    
    
    Parameters
    ----------
    da_ g: DataArray
        The `DataArray` object that has tiles of a g-point variable
        Tiles of the must be in their original llc layout.

       
    Returns
    -------
    da_g_new: DataArray
        a new `DataArray` object that has the appended values of 'g' along
        its top and right edges. 
    """
    
    #%%
    # the i_g and j_g dimensions will be incremented by one.
    llcN = len(da_g.i_g)
    j_g = np.arange(1, len(da_g.j_g)+2)
    i_g = np.arange(1, len(da_g.i_g)+2)


    # the k dimension, if it exists, is unchanged.
    if 'k' in da_g.dims:
        nk = len(da_g.k)  
        k = da_g['k'].values
    else:
        nk = 0

    # the time dimension, if it exists, is unchanged
    if 'time' in da_g.dims:
        time = da_g['time'].values
            
    #%%
    #print "\n>>> ADDING BORDERS TO G POINT TILES\n"
    
    # the new arrays will be one longer in each dimensin
    pad_i = 1 # pad by one in y (one new row)
    pad_j = 1 # do not pad in x 

    # set the number of processed tiles counter to zero    
    num_proc_tiles = 0

    # find the number of non-tile dimensions
    if 'tile' in da_g.dims:
        num_dims = da_g.ndim - 1
    else:
        num_dims = da_g.ndim

    if len(da_g.tile.values) == 1:
        raise ValueError('You can not append to da_g because you only \
                          passed a single tile!')

    # loop through all tiles in da_g
    for tile_index in da_g.tile.values:
        
        # find out which tiles are adjacent 
        right_tile_index, top_tile_index, corner_tile_index = \
            get_llc_tile_border_mapping(tile_index)
        
        print ('ti: ', tile_index, ' rti, tti, cti', right_tile_index, top_tile_index, corner_tile_index)
        ref_arr = deepcopy(da_g.sel(tile=tile_index))

        # the append_border flag will be true if we have a tile to the top.
        append_border_top    = False
        append_border_right  = False 
        append_border_corner = False         
        
        #print '\ncurrent da_g tile ', tile_index
        #print 'top tile index ', top_tile_index
        #print 'right tile index ', right_tile_index
        #print 'corner tile index ', corner_tile_index
        
        top_arr = []
        right_arr = []
        corner_arr = []
 

        if top_tile_index in da_g.tile.values:
            #print 'we have da_g tile to the top ', top_tile_index                      
            top_arr = da_g.sel(tile=top_tile_index)
            append_border_top = True

        if right_tile_index in da_g.tile.values:
            #print 'we have da_g tile to the right ', right_tile_index                      
            right_arr = da_g.sel(tile=right_tile_index)
            append_border_right = True
            
        if corner_tile_index in da_g.tile.values:
            #print 'we have da_g tile in the NE corner ', corner_tile_index                      
            corner_arr = da_g.sel(tile=corner_tile_index)
            append_border_corner = True

        print ('abr, abt, abc ', append_border_right, append_border_top, append_border_corner)
        
        print (type(top_arr), type(right_arr), type(corner_arr))
        
        if append_border_top or append_border_right or append_border_corner:
            new_arr=append_border_to_tile(ref_arr, tile_index,
                                          'g', llcN,
                                          top = top_arr,
                                          right = right_arr,
                                          corner = corner_arr)

        # if not then we will append an array of nans
        else:
            if num_dims == 2:
                pad = ((0, pad_i), (0, pad_j))
            elif num_dims == 3:
                pad = ((0, 0),     (0, pad_i), (0, pad_j))
            elif num_dims == 4:
                pad = ((0, 0),     (0, 0),     (0, pad_i), (0, pad_j))
            
            new_arr = np.pad(ref_arr, pad_width = pad, mode='constant',
                             constant_values = np.nan)

        # create a new DataArray
        if num_dims == 2:
            new_coords = [('j_g', j_g), ('i_g', i_g)]
        elif num_dims == 3 and nk > 0:
            new_coords = [('k', k), ('j_g', j_g), ('i_g', i_g)]
        elif num_dims == 3 and nk == 0:
            new_coords = [('time', time),('j_g', j_g), ('i_g', i_g)]
        elif num_dims == 4:
            new_coords = [('time', time), ('k', k), ('j_g', j_g), ('i_g', i_g)]
            
        tmp_DA = xr.DataArray(new_arr, name = da_g.name, coords=new_coords)

        # give the new DataArray the same attributes as da_g
        tmp_DA.attrs = da_g.attrs

        # give the new DataArray a tile coordinate
        tmp_DA.coords['tile'] = tile_index
    
        # increment the number of processed tiles counter by one
        num_proc_tiles += 1
        
        # set da_g_new equal to tmp_DA if this is the first processed tile
        if num_proc_tiles == 1:
            da_g_new = tmp_DA
        # otherwise, concatentate tmp_DA with da_g_new along the 'tile' dim
        else:
            da_g_new = xr.concat([da_g_new, tmp_DA],'tile')

        # reset tmp_DA
        tmp_DA = []

    # add all time coordinates to_da_g_new from da_g.
    for idx, var in enumerate(da_g.coords):
        if 'tim' in var:
            da_g_new[var] = da_g[var]

    da_g_new.attrs['padded'] = True            
            
    #%%            
    return da_g_new
    #%%
    

def get_llc_tile_border_mapping(tile_index):
    """
    Return the tile numbers for the tiles adjacent to the tile at 
    `tile_index` along the 'right','top' and 'corner' in the original llc tile layout (if it exists)
    
    Note
    ----
        In some cases there is no tile in the adjacent location. 
        For example, there is no tile to the "right" of tile index 12 
        in the original llc tile layout.
        
        Also, uses tile index (0..12) not tile name (which could be 1..13 or 
        0..12) or something else 
        
    Parameters
    ----------
    tile_index : int
        the index of the reference tile [0-12].
    

    Returns
    -------
    right_tile_index, top_tile_index, corner_tile_index : ints
        the indices of the tiles to the right, top, and top or bottom right 
        corner of `tile_index` (bottom right for tiles 4 and 5).  
        If there is no tile in the location, -1 is returned.
        

    """
    if tile_index == 0:
        right_tile_index = 3
        top_tile_index = 1
        corner_tile_index= 4
    elif tile_index == 1:
        right_tile_index = 4
        top_tile_index = 2
        corner_tile_index = 5
    elif tile_index == 2:
        right_tile_index = 5
        top_tile_index = 6
        corner_tile_index = 6 
    elif tile_index == 3:
        right_tile_index = 9
        top_tile_index = 4
        corner_tile_index = 9 
    elif tile_index == 4:
        right_tile_index = 8
        top_tile_index = 5
        corner_tile_index = 9 # bottom right ccorner
    elif tile_index == 5:
        right_tile_index = 7
        top_tile_index = 6
        corner_tile_index = 8  # bottom right corner
    elif tile_index == 6:
        right_tile_index = 7
        top_tile_index = 10
        corner_tile_index = 10 
    elif tile_index == 7:
        right_tile_index = 8
        top_tile_index = 10
        corner_tile_index = 11
    elif tile_index == 8:
        right_tile_index = 9
        top_tile_index = 11
        corner_tile_index = 12
    elif tile_index == 9:
        right_tile_index = -1 # does not exist 
        top_tile_index = 12
        corner_tile_index = -1 # does not exist 
    elif tile_index == 10:
        right_tile_index = 11
        top_tile_index = 2
        corner_tile_index = -1 # does not exist.
    elif tile_index == 11:
        right_tile_index = 12
        top_tile_index = 1
        corner_tile_index = 2 # yes, corner is "top left"
    elif tile_index == 12:
        right_tile_index = -1 # does not exist 
        top_tile_index = 0
        corner_tile_index = 1 # yes, corner is "top left"
    
    return right_tile_index, top_tile_index, corner_tile_index   
 
            
def add_borders_to_GRID_tiles(gds):
    """
    A routine that appends the partial halo for all grid tile variables.  
    
    Parameters
    ----------
    gds : Dataset
        the `Dataset` object that has one or more grid ECCO v4 tile.
        The tiles of the grid Dataset must be in their original layout.
    

    Returns
    -------
    gds_pad : Dataset
        a new `Dataset` object for the ECCO v4 grid that has been padded
        on the top and right sides for 'v' and 'u' point variables, 
        respectively and on the top, top right corner, and right sides for
        'g' point variables.  'c' point and vertical grid parameters are
        untouched.
        
        gds_pad also has the new attribute, padded=True
    """            
    #%%
    print("\n>>> ADDING BORDERS TO GRID TILES\n")

    # C points
    ## no borders added to C points.  We're adding borders AROUND c points

    print(" --- 'g' points")
    # G points
    XG_b  = add_borders_to_DataArray_G_points(gds.XG).to_dataset();

    YG_b  = add_borders_to_DataArray_G_points(gds.YG).to_dataset();    

    RAZ_b = add_borders_to_DataArray_G_points(gds.RAZ).to_dataset();        

    print(" --- 'u' points")
    # U points
    DXC_b    = add_borders_to_DataArray_U_points(gds.DXC, 
                                                 gds.DYC).to_dataset()

    DYG_b    = add_borders_to_DataArray_U_points(gds.DYG, 
                                                 gds.DXG).to_dataset()

    hFacW_b  = add_borders_to_DataArray_U_points(gds.hFacW, 
                                                 gds.hFacS).to_dataset()

    land_u_b = add_borders_to_DataArray_U_points(gds.land_u, 
                                                 gds.land_v).to_dataset()   
  
    print(" --- 'v' points")
    # V points
    DYC_b    = add_borders_to_DataArray_V_points(gds.DXC, 
                                                 gds.DYC).to_dataset()

    DXG_b    = add_borders_to_DataArray_V_points(gds.DYG, 
                                                 gds.DXG).to_dataset()

    hFacS_b  = add_borders_to_DataArray_V_points(gds.hFacW, 
                                                 gds.hFacS).to_dataset()

    land_v_b = add_borders_to_DataArray_V_points(gds.land_u, 
                                                 gds.land_v).to_dataset()   
    # now merge 
    
    # pull off the vertical parameters
    gds_tmp_Z = gds[['RC','RF','DRC','DRF']]
    
    # drop all of the parameters that we just added borders to and the vertical
    # grid parameters
    gds_tmp = gds.drop(['DYC','DXG', 'hFacS',
                        'DXC', 'DYG','hFacW','XG','YG',
                        'RAZ','RC','RF','DRC','DRF'])
    
    # merge the old c-point grid parameters and the new padded 'g','u', and 'v'
    # point parameters, and the vertical grid parameters
    gds_pad = xr.merge([gds_tmp, XG_b, YG_b, RAZ_b, DXC_b, DYG_b, hFacW_b, 
                        land_u_b, DYC_b, DXG_b, hFacS_b, land_v_b, gds_tmp_Z])
    
    # give the new grid Dataset the same parameters
    gds_pad.attrs = gds.attrs
    gds_pad.attrs['padded'] = True
    
    #%%
    return gds_pad
    #%%
