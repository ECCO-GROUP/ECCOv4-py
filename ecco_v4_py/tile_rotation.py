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

        

def reorient_13_tile_GRID_Dataset_to_latlon_layout(gds, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # By default we will keep the Arctic tile aligned with Tile 6.
    aca = 6

    # However, an optional argument allows the alignment to be changed.
    for key in kwargs:
        if key == "Arctic_Align":
            aca = kwargs[key]
    
    #print 'ACA: ', aca
    
    # ROTATION PART 1: GRID PARAMETERS ON THE 'C' AND 'G' POINTS
    # -- step 1: select the grid parameters that are situated on the 
    #            'C' or 'G' points.  These 'rotate' easily because these 
    #            variables have no 'direction' (XC after rotation is still XC)
    #            (except maybe the Angles but haven't sorted those out yet)
    ds_CG = gds[['XC','YC', 'Depth','hFacC', 'RAC', 
                 'AngleCS', 'AngleSN','XG','YG','RAZ', 'land_c']]

    # -- step 2: rotate the new subsetted Dataset 

    ds_CG_r = reorient_13_tile_Dataset_to_latlon_layout_CG_points(ds_CG, **kwargs)

    for key, value in ds_CG_r.variables.iteritems():
        if key not in ds_CG_r.coords:
            ds_CG_r.variables[key].attrs['Arctic_Align'] = aca

    # create two Datasets with subsets of grid parameters situated on 
    # 'U' and 'V' points, respectively.  Rotation of 'U' and 'V' point 
    # variables to the lat-lon orientation requires that the order of the 
    # variables in the two Datasets are 
    # 
    # correct order (e.g., after rotation DXC on tile 6 corresponds with DYC 
    # in tile 8, DYG on tile 6 corresponds with DXG on tile 8 after rotation)
    
    ds_U = gds[['DXC','DYG','hFacW','land_u']]
    ds_V = gds[['DYC','DXG','hFacS','land_v']]    
    
    ds_U_r, ds_V_r = \
        reorient_13_tile_Dataset_to_latlon_layout_UV_points(ds_U, ds_V, **kwargs)

    # merge the rotated Datasets on the C, G, U and V points.
    gds_r = xr.merge([ds_CG_r, ds_U_r, ds_V_r])
    
    for key, value in gds_r.variables.iteritems():
        if key not in gds_r.coords:
            gds_r.variables[key].attrs['Arctic_Align'] = aca
            gds_r.variables[key].attrs['grid_layout'] = 'rotated llc' 

    #%%
    gds_r.attrs = gds.attrs
    return gds_r
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




def reorient_13_tile_Dataset_to_latlon_layout_CG_points(ds, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # this routine takes a Dataset with multiple llc tiles containing
    # variables on the 'C' or 'G' points of the Arakawa-C grid and, if 
    # present, rotates tiles 8-13 so that they line up with tiles 1-6 
    # in a quasi lat-lon layout.  
    #
    # note: the Dataset can be a subset of the 13 llc tiles.
    # 
    # The Actic Cap Tile 7 can be rotated in one of four ways to align it 
    # with tiles 3, 6, 8, or 11.  The default orientation of tile 7 has it 
    # aligned with tile 6 in the 'y' direction and tile 8 in the 'x' direction
    # Tile 7's alignment in the 'y' direction can be changed to a different 
    # tile using the 'Arctic_Align' argument.  
    #
    # For example, to make tile 7 line up with tile 3 in the 'y' direction use
    # the argument Arctic_Align = 3
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    print "\n>>> REORIENTING DATASET TO LATLON LAYOUT, C OR G POINTS"
    # create an empty array that will contain the tiles after rotation
    ds_all = []
    # save the metadata from the Dataset, we'll use it later.
    attrs = ds.attrs
    
    # By default, the Arctic cap will remain aligned with tile 6 in 
    # the 'y' direction
    aca = 6
    
    #%%
    # To make the Arctic cap align with a different tile in the 'y' direction
    # specify the option 'Artic_Cap_Alignment' argument
    for key in kwargs:
        if key == "Arctic_Align":
            aca = kwargs[key]


    #%%            
    # counter for the number of tiles that we've processed
    num_tiles = 0
    
    # loop through all 13 tiles
    for cur_tile in range(1,14):

        # Proceed if cur_tile is in the Dataset
        if cur_tile in ds.tile.values:
            # increment the number of tiles found.
            num_tiles = num_tiles + 1

            # extract the Dataset corresponding with this tile
            cur_ds = ds.sel(tile=cur_tile)
        
            # For tiles 1-6 we do nothing, they are already in a quasi lat-lon 
            # orientation
            
            # Tile 7, the Arctic Cap, can be rotated to align with tile
            # 3, 6, 8, or 11 in the 'y' direction.
            # If the Arctic is be aligned with a different tile it must be rotated
            # by 90 degrees clockwise more than one time.
            if cur_tile == 7:
                if aca == 3:
                    cur_ds = rotate_single_tile_Dataset_CG_points(cur_ds, 
                                                                  rot_k =3)
                elif aca == 6:
                    print 'keeping Arctic cap alignment with tile 6'
                elif aca == 8:                
                    cur_ds = rotate_single_tile_Dataset_CG_points(cur_ds)                
                elif aca == 11:
                    cur_ds = rotate_single_tile_Dataset_CG_points(cur_ds, 
                                                                  rot_k=2)
                else:
                    print 'invalid Arctic cap alignment, leaving unchanged'
    
            # Tiles 8-13 are rotated once clockwise
            elif cur_tile > 7:
                cur_ds = rotate_single_tile_Dataset_CG_points(cur_ds)                
    
            # Append the new, possibly rotated Dataset, to the new Dataset array
            if num_tiles == 1:
                ds_all = cur_ds
            else:
                ds_all = xr.concat([ds_all, cur_ds],'tile')

        else:
            print 'Tile not present in Dataset ', cur_tile
            
    # Give the new Dataset the same metadata as the original Dataset.
    ds_all.attrs = attrs

    for key, value in ds_all.variables.iteritems():
        if key not in ds_all.coords:
            ds_all.variables[key].attrs['Arctic_Align'] = aca 
            ds_all.variables[key].attrs['grid_layout'] = 'rotated llc'
            
    #%%
    return ds_all


#%%
def rotate_single_tile_Dataset_CG_points(ds, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotates a set of variables in a single Dataset tile
    # by some multiple of 90 degrees in the horizontal plane.  
    #
    # The number of 90 degree rotations is specified by the optional 
    # 'rot_k' argument
    #
    # The variables in this Dataset must all be on the 
    # 'C' or 'G' points of the Arakawa-C grid.  
    #
    # Horizontal dimensions must be the same in both x and y directions.
    #
    # Note: rotation of variables on the 'U' or 'V' points of the Arakawa-C 
    # grid require special attention and use a different subroutine.
    # 
    # One use of this routine is to rotate tiles 8-13 from the original
    # llc tile layout to match the quasi lat-lon layout of tiles 1-6.
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # First do a little sanity check - if the Dataset has more than one tile 
    # the routine will blindly rotate them all by 90 degrees.  This is dangerous
    # since normally we only want to rotate tiles 7-13 and even then by 
    # different multiples of 90 degrees.
    if 'tile' in ds.coords:
        if ds.tile.size > 1:
            print 'WARNING: you are passing a Dataset with more than one tile to "rotate_single_tile_Dataset_CG_points"'
            
        
    # loop through each variable in the Dataset and select out for rotation 
    # only those variables that are not coordinates
    for key, value in ds.variables.iteritems():
        if key not in ds.coords:
            
            # pull out the variable from the Dataset.  A single variable from
            # the Dataset is a DataArray
            da = ds[key]
            
            # Send the DataArray off to be rotated
            ds[key].values = \
                rotate_single_tile_DataArray_CG_points(da, **kwargs)

            # Assign this new attribute to the variable
            ds[key].attrs['grid_layout'] = 'rotated llc'

    #%%                         
    return ds

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def rotate_single_tile_DataArray_CG_points(da, **kwargs):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotates a variable from a single DataArray tile
    # by some multiple of 90 degrees in the horizontal plane.  
    #
    # The number of 90 degree rotations is specified by the optional 
    # 'rot_k' argument
    #
    # The variable in this DataArray must be on the 
    # 'C' or 'G' points of the Arakawa-C grid.  
    #
    # Note: rotation of variables on the 'U' or 'V' points of the Arakawa-C 
    # grid require special attention and use a different subroutine.
    # 
    # One use of this routine is to rotate tiles 8-13 from the original
    # llc tile layout to match the quasi lat-lon layout of tiles 1-6.
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # First a sanity check to see if the Dataset has more than one tile.
    if 'tile' in da.coords:
        if da.tile.size > 1:
            print 'WARNING: you are passing a DataArray with more than one tile to "rotate_single_tile_DataArray_CG_points"'

    # by default the array will be rotated clockwise 90 degrees in its 
    # horizontal dimensions rot_k = 1 times.  
    rot_k = 1
    
    # the number of 90 degree rotations can be specified with the optional 
    # 'rot_k' argument
    for key in kwargs:
        if key == "rot_k":
            rot_k = kwargs[key]
    
    #%%
    # get the dimensions of the variable
    num_dims = da.ndim

    # the rotation command depends on how many dimensions the variable
    # has.  The horizontal dimensions are always the last two axes.            
    if num_dims == 2:
        # assuming that the two dimensions are y and x                    
        da.values = np.rot90(da.values, k=rot_k)

    elif num_dims == 3:
        # assuming that the last two dimensions are y and x
        da.values = np.rot90(da.values,  k=rot_k , axes=(1,2))
 
    elif num_dims == 4:
        # assuming that the last two dimensions are y and x                    
        da.values = np.rot90(da.values,  k=rot_k , axes=(2,3))

    else:
        print 'variables must have 2, 3, or 4 dimensions'
        print da.name + ' has ' + str(num_dims) + ' dimensions\n'
        print 'so I am returning ' + da.name + ' unchanged.'
        
    # give this variable a new attribute
    da.attrs['grid_layout'] = 'rotated_llc'
    #%%            
    return da.values
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        



def reorient_13_tile_Dataset_to_latlon_layout_UV_points(ds_U, ds_V, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Take a pair of Datasets containing variables 
    # on the 'U' or 'V' points of the Arakawa-C grid and rotate 
    # those variables in tiles 8-13 so that they line up with tiles 1-6 
    # in a quasi lat-lon layout.  
    #
    # The Arctic Cap (tile 7) can be rotated so 
    # that it aligns with tiles 3, 6, 8, or 11 in the 'y' direction.
    # The default orientation of tile 7 has it 
    # aligned with tile 6 in the 'y' direction and tile 8 in the 'x' direction.
    #
    # To change the alignment of tile 7 in the 'y' direction to a different
    # tile specify the target tile with the 'Arctic_Align' argument.  
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # create an empty array that will contain the tiles after rotation
    ds_U_all = []
    ds_V_all = []    

    # save the metadata from the Dataset, we'll use it later.
    U_attrs = ds_U.attrs
    V_attrs = ds_V.attrs 
    
    # By default, the Arctic cap will remain aligned with tile 6 in 
    # the 'y' direction
    aca = 6
    
    #%%
    # To make the Arctic cap align with a different tile in the 'y' direction
    # specify the option 'Artic_Cap_Alignment' argument
    for key in kwargs:
        if key == "Arctic_Align":
            aca = kwargs[key]

    #print "Reorient UV ACA: ", aca
    #%%            
    # counter for the number of tiles that we've processed
    num_tiles = 0
    
    # loop through all 13 tiles
    for cur_tile in range(1,14):

        #print cur_tile

        # Check to make sure cur_tile is in the ds_U Dataset (if it is assume
        # is it also in ds_V)
        if cur_tile in ds_U.tile.values:
            #print 'TILE: ', cur_tile
            # increment the number of tiles found.
            num_tiles = num_tiles + 1

            # extract the Dataset corresponding with this tile
            cur_ds_U = ds_U.sel(tile=cur_tile)
            cur_ds_V = ds_V.sel(tile=cur_tile)
            
            # For tiles 1-6 we do nothing, they are already in a quasi lat-lon 
            # orientation
            
            # Tile 7, the Arctic Cap, can be rotated to align with tile
            # 3, 6, 8, or 11 in the 'y' direction.
            # If the Arctic is be aligned with a different tile it must be 
            # rotated by 90 degrees clockwise more than one time.
            if cur_tile == 7:
                if aca == 3:
                    cur_ds_U, cur_ds_V = \
                        rotate_single_tile_Datasets_UV_points(cur_ds_U, 
                                                              cur_ds_V,
                                                              rot_k=3)
                elif aca == 6:
                    print 'keeping Arctic cap alignment with tile 6'
                elif aca == 8:                
                    cur_ds_U, cur_ds_V = \
                        rotate_single_tile_Datasets_UV_points(cur_ds_U, 
                                                              cur_ds_V)
                elif aca == 11:
                    cur_ds_U, cur_ds_V = \
                        rotate_single_tile_Datasets_UV_points(cur_ds_U, 
                                                              cur_ds_V,
                                                              rot_k=2)
                else:
                    print 'invalid Arctic cap alignment, leaving unchanged'
    
            # Tiles 8-13 are rotated once clockwise
            elif cur_tile > 7:
                cur_ds_U, cur_ds_V = \
                    rotate_single_tile_Datasets_UV_points(cur_ds_U, 
                                                          cur_ds_V,)                
    
            # Append the new Dataset to the new Dataset array
            if num_tiles == 1:
                ds_U_all = cur_ds_U
                ds_V_all = cur_ds_V
            else:
                ds_U_all = xr.concat([ds_U_all, cur_ds_U],'tile')
                ds_V_all = xr.concat([ds_V_all, cur_ds_V],'tile')            
        else:
            print "Tile not included in Dataset ", cur_tile
            
    # Give the new Dataset the same metadata as the original Dataset.
    ds_U_all.attrs = U_attrs
    ds_V_all.attrs = V_attrs 
    #%%
    
    for key, value in ds_U_all.variables.iteritems():
        if key not in ds_U_all.coords:
            ds_U_all.variables[key].attrs['Arctic_Align'] = aca
            ds_U_all.variables[key].attrs['grid_layout'] = 'rotated llc' 

    for key, value in ds_V_all.variables.iteritems():
        if key not in ds_V_all.coords:
            ds_V_all.variables[key].attrs['Arctic_Align'] = aca
            ds_V_all.variables[key].attrs['grid_layout'] = 'rotated llc'
            
    return ds_U_all, ds_V_all



#%%
def rotate_single_tile_Datasets_UV_points(ds_U, ds_V, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotates a set of variables in two Datasets for a single tile 
    # by some multiple of 90 degrees in the horizontal plane.  
    #
    # The number of 90 degree rotations is specified by the optional 
    # 'rot_k' argument
    #
    # The variables in Dataset ds_U must all be on the 'U' points and 
    # the variables in Dataset ds_V must all be on the 'V' points 
    # of the Arakawa-C grid.  
    #
    # One use of this routine is to rotate tiles 8-13 from the original
    # llc tile layout to match the quasi lat-lon layout of tiles 1-6.
    #
    # Corresponding U and V point variables must be provided in the identical
    # order in the Datasets. e.g., DXC corresponds with DYC because the 
    # the direction of X and Y change after an odd number of rotations.
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # First a sanity check - if the Dataset has more than one tile 
    # the routine will blindly rotate them all by 90 degrees. This is dangerous
    # since normally we only want to rotate tiles 7-13 and even then by 
    # different multiples of 90 degrees.
    if 'tile' in ds_U.coords:
        if ds_U.tile.size > 1:
            print 'WARNING: you are passing a Dataset with more than one tile'
            
        
    # loop through each variable in the Dataset and select out for rotation 
    # only those variables that are not coordinates
    
    print '\nROTATING TILE ', ds_U.tile.values
    
    num_vars = 0
    ki = 0
    for key, value in ds_U.variables.iteritems():
        ki = ki+1
        if key not in ds_U.coords:
            
            num_vars = num_vars+1
            
            cur_V_key = ds_V.variables.keys()[ki-1]
            cur_U_key = ds_U.variables.keys()[ki-1]

            print 'PAIRING ' + cur_U_key + ' with ' + cur_V_key
            
            da_U = ds_U[cur_U_key]
            da_V = ds_V[cur_V_key]
 
            # Send the DataArray off to be rotated
            da_Ur, da_Vr = \
                rotate_single_tile_DataArrays_UV_points(da_U, da_V, **kwargs)

            ds_U[cur_U_key].values = da_Ur.values
            ds_V[cur_V_key].values = da_Vr.values
            
            ds_U[cur_U_key].attrs = da_Ur.attrs
            ds_V[cur_V_key].attrs = da_Vr.attrs
            
            
    #%%                         
    return ds_U, ds_V

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def rotate_single_tile_DataArrays_UV_points(da_U, da_V, **kwargs):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Rotate a pair of DataArray variables for a single tile by some
    # multiple of 90 degrees.  
    # da_U is a variable on the 'U' points of 
    # the tile while da_V is the variable on the 'V' points of the tile.  
    #
    # Note, DataArrays are essentially single variable versions of Datasets.
    #   da_U = ds['U point variable name']
    #   da_V = ds['V point variable name']    
    #
    # The DataArrays can have 2, 3, or 4 dimensions.
    #
    # If the number of 90 degree rotations is odd, the variable that was 
    # originally on the 'U' points will be situated on the 'V' points and 
    # vice versa because the meaning of x and y will be swapped.  The 
    # rotated version of U will come from V and vice versa.  
    # On the other hand, if the 
    # number of 90 degree rotations is even the rotated version of U comes 
    # from U and vice versa.
    #
    # If the variables involve velocity or fluxes, the variables may need
    # to change sign after rotation. 
    # For example, consider rotation of velocity variables by 90 degrees 
    # in tiles 8-13 in the original llc tile layout.
    # After rotation, 
    # +U velocities in the +x direction of tiles 8-13 become -V velocties 
    # in the y direction after rotation (sign change) 
    # +V velocities in the +y direction of tiles 8-13 become +U velocities
    # in the x direction after rotation (no sign change)
    #
    # If you want the signs to be flipped in either U or V, specify with
    # the u_sign_flip or v_sign_flip parameters.  To flip the
    # sign of 'U' variables in tile 8 after rotating them by 90 degrees:
    # >da_U = ds['UVEL'].sel(tile=8)
    # >da_V = ds['VVEL'].sel(tile=8)    
    # >da_Ur, da_Vr = 
    #     rotate_single_tile_DataArrays_UV_points(da_U, da_V, 
    #                                             u_sign_flip=True):
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # by default the array will be rotated clockwise 90 degrees in its 
    # horizontal dimensions rot_k = 1 times.  
    rot_k = 1
    
    #%%
    # the number of 90 degree rotations can be specified with the optional 
    # 'rot_k' argument
    for key in kwargs:
        if key == "rot_k":
            rot_k = kwargs[key]

    #%%
    # By default we do not flip the sign of U and V
    flip_U = False
    flip_V = False
    #%%
    # look for the u_sign_flip and the v_sign_flip arguments
    for key in kwargs:
        if key == "u_sign_flip" and kwargs[key] == True:
            flip_U = True

        elif key == "v_sign_flip" and kwargs[key] == True:
            flip_V = True

    #%%
    if flip_U:
        print "Flipping the sign of the variable on the 'U' point \n"
        da_U = da_U * -1
    if flip_V:
        print "Flipping the sign of the variable on the 'V' point \n"
        da_V = da_V * -1

    da_U_new = deepcopy(da_U)
    da_V_new = deepcopy(da_V)  
       
    # in the special case where the number of rotations is even 
    # then the array dimensions do not change and the rotated version of
    # da_U comes from da_U and the rotated version of da_V comes from da_V
    if np.mod(rot_k, 2) == 0:
        print "Even number of rotations\n"
        
        da_U_new[:] = np.rot90(da_U, k=rot_k, axes=(-2,-1))
        da_V_new[:] = np.rot90(da_V, k=rot_k, axes=(-2,-1))
        
    else:
        rot_U = np.rot90(deepcopy(da_U), k=rot_k, axes=(-2,-1))
        rot_V = np.rot90(deepcopy(da_V), k=rot_k, axes=(-2,-1))
        
        da_U_new[:] = rot_V
        da_V_new[:] = rot_U
      
    da_U_new.attrs['grid_layout'] = 'rotated llc' 
    da_V_new.attrs['grid_layout'] = 'rotated llc'
        
    print da_U_new.attrs
    print da_V_new.attrs
    
    #%%   
    return da_U_new, da_V_new
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
           
