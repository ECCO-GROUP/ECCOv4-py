#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:11:15 2017

@author: ifenty
"""
from __future__ import division,print_function
import numpy as np
import xarray as xr
import time
from copy import deepcopy
import glob

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def load_subset_tiles_from_netcdf(data_dir, var, var_type, 
                                  tile_subset, 
                                  k_subset = [],
                                  keep_landmask_and_area = False,
                                  less_output = False):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Reads and concatenates a subset of between 1 and 13 netcdf ecco tiles 
    # as an xarray DataSet object.  
    # A new dimension is added, 'i0', corresponding to 
    # tile number (1, 2, ... 13)
    # * can handle 2D output: lat, lon [e.g., a time-mean surface field]    
    # * can handle 3D output: lat, lon, time [e.g., SSH]
    # * can handle 4D output: lat, lon, depth, time [e.g., THETA]    
    #
    # tiles are kept in their original llc orientation (i.e., tiles 8-13 are
    # rotated by 90 relative to tiles 1-6).
    #
    # data_dir : the directory with the netcdf files
    #    
    # var      : name of the variable in the netcdf tile file.  For example, 
    #            var is 'GRID' for the grid files [GRID.0001.nc, GRID.0002.nc, ...
    #            GRID.0013.nc] and var is 'THETA' for theta files [TEHTA.0001.nc
    #            THETA.0002.nc, ... THETA.00013.nc]
    #
    # var_type : variables can be of one of 5 types.
    #            'c' for variables situated at the cell center [e.g., THETA, SALT, SSH]
    #            'g' for variables situated at the cell corners 
    #            'u' for variables situated at the cell u-velocity points [e.g., UVEL]
    #            'v' for variables situated at the cell v-velocity points [e.g., VVEL]
    #            'grid' - a special case for the netcdf GRID files that contain
    #                     llc grid parameters variables.  GRID files contain
    #                     variables situated on a mix of 'c','g','u', and 'v' 
    #                     points.  
    #
    # var_type also determines how the coordinates of the variable are renamed.
    # The default variable dimension names in the llc tile files are not very
    # descriptive (e.g., 'i1','i2','i3').  
    # The coordinates corresponding with horizontal dimensions are renamed 
    # depending on var_type.  To wit,
    #  'i'   and 'j'   for 'c' point variables 
    #  'i_g' and 'j_g' for 'g' point  variables
    #  'i_g' and 'j'   for 'u' point variables
    #  'i'   and 'j_g' for 'v' point variables
    #
    # subset : a list of tiles to load
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # By default print messages to the screen when loading tile files

    start = time.time()

    num_loaded_tiles = 0

    for tile_index in range(1, 14):  
        if tile_index in tile_subset:
            ds_tile = load_tile_from_netcdf(data_dir, var, var_type, 
                        tile_index,
                        k_subset = k_subset,
                        keep_landmask_and_area = keep_landmask_and_area,
                        less_output = less_output)

            num_loaded_tiles += 1

            if num_loaded_tiles == 1:
                #ds = {str(tile_index):ds_tile}
                ds = ds_tile
            else:
                #ds[str(tile_index)]=ds_tile
                ds = xr.concat((ds, ds_tile),'tile')

            ds_tile = []
        else:
            if less_output == False:
                print('skipping this tile, not on the list ', tile_index)
                
    end = time.time()

    if less_output == False:
        print('total file load and concat time ', end-start, 's')

    return ds


#%%
def load_all_tiles_from_netcdf(data_dir, var, var_type, 
                               k_subset = [],
                               keep_landmask_and_area = False,
                               less_output = False):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Reads and concatenates all of the 13 netcdf ecco tiles as an xarray 
    # DataSet object.  A new dimension is added, 'i0', corresponding to 
    # tile number (1, 2, ... 13)
    # * can handle 2D output: lat, lon [e.g., a time-mean surface field]    
    # * can handle 3D output: lat, lon, time [e.g., SSH]
    # * can handle 4D output: lat, lon, depth, time [e.g., THETA]    
    #
    # tiles are kept in their original llc orientation (i.e., tiles 8-13 are
    # rotated by 90 relative to tiles 1-6).
    #
    # data_dir : the directory with the netcdf files
    #    
    # var      : name of the variable in the netcdf tile file.  For example, 
    #            var is 'GRID' for the grid files [GRID.0001.nc, GRID.0002.nc, ...
    #            GRID.0013.nc] and var is 'THETA' for theta files [TEHTA.0001.nc
    #            THETA.0002.nc, ... THETA.00013.nc]
    #
    # var_type : variables can be of one of 5 types.
    #            'c' for variables situated at the cell center [e.g., THETA, SALT, SSH]
    #            'g' for variables situated at the cell corners 
    #            'u' for variables situated at the cell u-velocity points [e.g., UVEL]
    #            'v' for variables situated at the cell v-velocity points [e.g., VVEL]
    #            'grid' - a special case for the netcdf GRID files that contain
    #                     llc grid parameters variables.  GRID files contain
    #                     variables situated on a mix of 'c','g','u', and 'v' 
    #                     points.  
    #
    # var_type also determines how the coordinates of the variable are renamed.
    # The default variable dimension names in the llc tile files are not very
    # descriptive (e.g., 'i1','i2','i3').  
    # The coordinates corresponding with horizontal dimensions are renamed 
    # depending on var_type.  To wit,
    #  'i'   and 'j'   for 'c' point variables 
    #  'i_g' and 'j_g' for 'g' point  variables
    #  'i_g' and 'j'   for 'u' point variables
    #  'i'   and 'j_g' for 'v' point variables
    #
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # By default print messages to the screen when loading tile files
    
    if less_output == False:
        print("\n>>> LOADING TILES FROM NETCDF\n")
    
    tile_subset = range(1,14)

    ds = load_subset_tiles_from_netcdf(data_dir, var, var_type, tile_subset,
                           k_subset = k_subset,
                           keep_landmask_and_area = keep_landmask_and_area,
                           less_output = less_output)

        
    #%%
    if var == 'GRID':
        # since we have concatenated 14 tiles along the 'tile' dimension
        # the vertical grid parameters now have a tile dimension.  This
        # is unncessary since vertical grid parameters are the same across
        # all tiles.  The code below drops the 'tile' dimension from the
        # vertical coordinates.
        
        # save the metadata on ds, we'll use it later.
        orig_attrs = ds.attrs

        # Go through each of these 1D coordinates and drop the 'tile'
        # dimension        
        #RB = ds.RB[0,:].drop('tile')
        RC = ds.RC[0,:].drop('tile')
        RF = ds.RF[0,:].drop('tile')
        DRC = ds.DRC[0,:].drop('tile')
        DRF = ds.DRF[0,:].drop('tile')
        
        # drop these coordinates from ds
        #ds = ds.drop(['RB', 'RC','RF','DRC','DRF']); 
        ds = ds.drop(['RC','RF','DRC','DRF']); 
        
        # add the coordinates back in now that we have dropped the 'tile'
        # dimension
        #ds = xr.merge([ds, RB, RC, RF, DRC, DRF])
        ds = xr.merge([ds, RC, RF, DRC, DRF])
        
        # reset the metadata on ds.
        ds.attrs = orig_attrs

    print("Finished loading all 13 tiles of ", var) 

    return ds
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
        
def load_tile_from_netcdf(data_dir, var, var_type, tile_index, 
                          k_subset = [],
                          keep_landmask_and_area = False,
                          less_output = False):

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    # Reads one of the 13 netcdf ecco tiles as an xarray DataSet object
    # Tiles 1-6, and 8-13 are rotated so that they align in lat/lon 
    # Tile 7 (Arctic Cap)  is rotated to line up with Tile 11 (N. America)
    # * can handle 2D output: lat, lon, time [e.g., SSH]
    # * can handle 3D output: lat, lon, depth, time [e.g., THETA]
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # by default, we will drop the tile's record of land mask, grid cell
    # area, and lat and lon coordinates.  this information is redundant 
    # because the ECCO grid has all of this information.  keeping it around
    # just makes for very messy Datasets
    
    
    # construct the netcdf file name based on the variable name and the
    # the tile index
    fname = (data_dir + var + '.' + str(tile_index).zfill(4) + '.nc')

    if less_output == False:
        print('loading ', fname)

    # load the netcdf file using xarray.  
    # xarray automatically converts the netcdf file into a Dataset object
    # using the netcdf metadata

    # check to see if file exists.    
    file = glob.glob(fname)
    if len(file) == 0:
        raise IOError(fname + ' not found ')

    ds = xr.open_dataset(fname, decode_times=False)

    # finally, give this Dataset the 'tile' coordinate with current tile_index
    ds.coords['tile'] = tile_index
    # xarray doesn't do a perfect job of interpreting the information in the
    # netcdf files.  To improve unsability we need to make some changes to the
    # new Dataset object.
    #
    # first, change the type of all coordinate index variables from float64 to 
    # int64.  Coordinate index variables are discrete [1, 2, ... n] 
    # so integers are more logical than floats.
    # -- loop through all dimensions and change the type to int
    #for key, value in ds.dims.iteritems():
    #    ds[key] = ds[key].astype(np.int)

    # second, check to see if we are loading a GRID tile files.  GRID tiles
    # need special treatment because they are a mix of horizontal and vertical
    # parameters on a mix of Arakawa C-grid points ('c','g','u', and 'v')

    if var != 'GRID':
        # we do not have a GRID tile file, we have either a 4D file like THETA
        # which has four dimensions: time, depth, y, and x, or a 3D file like
        # or a 3D file like SSH which has three dimensions: time, y, and x
        num_dims = len(ds.dims)

        # the model time step 'timstep' is loaded as a variable instead of as
        # a coordinate label.  Here, I make 'timstep' a coordinate label.
        # -- step 1, add a corordinate to the Dataset with the name 'timestep'
        #            with the value of the original 'timstep' variable
        ds.coords['timestep'] = ('time', ds['timstep'].values)
        # -- step 2, drop 'timstep' as a variable from the Dataset
        ds = ds.drop('timstep')
        
        ds = ds.expand_dims('tile')

        # give the coordinates more meaningful names depending on where on the
        # Arakawa C-grid the variable is situated.
        if num_dims == 4 :
            if len(k_subset) > 0 :
                ds = ds.sel(i2=k_subset)

            if var_type == 'c':
                ds=ds.rename({'i1':'time','i2':'k', 'i3':'j','i4':'i'})                    
                ds = ds.transpose('time','tile','k','j','i')
                
            elif var_type == 'g':
                ds=ds.rename({'i1':'time','i2':'k', 'i3':'j_g','i4':'i_g'})                                    
                ds = ds.transpose('time','tile','k','j_g','i_g')
            elif var_type == 'u':
                ds=ds.rename({'i1':'time','i2':'k', 'i3':'j','i4':'i_g'})                                    
                ds = ds.transpose('time','tile','k','j','i_g')
                
            elif var_type == 'v':
                ds=ds.rename({'i1':'time','i2':'k', 'i3':'j_g','i4':'i'})                                    
                ds = ds.transpose('time','tile','k','j_g','i')
                
            else:
                print('to give sensible coordinate names \'var_type\' must be one of: c, g, u or v')

        # give the coordinates more meaningful names depending on where on the
        # Arakawa C-grid the variable is situated.
        elif num_dims == 3 :
            if var_type == 'c':
                ds = ds.rename({'i1':'time','i2':'j','i3':'i'})  
                ds = ds.transpose('time','tile','j','i')

            elif var_type == 'g':
                ds=ds.rename({'i1':'time','i2':'j_g','i3':'i_g'})                                    
                ds = ds.transpose('time','tile','j_g','i_g')
                
            elif var_type == 'u':
                ds=ds.rename({'i1':'time','i2':'j','i3':'i_g'})                                    
                ds = ds.transpose('time','tile','j','i_g')
                
            elif var_type == 'v':
                ds=ds.rename({'i1':'time','i2':'j_g','i3':'i'})     
                ds = ds.transpose('time','tile','j_g','i')
                
            else:
                print('to give sensible coordinate names \'var_type\' must be one of: c, g, u or v\n')
            
        # ECCO v4 Model output on the tile files include four auxillary fields,
        # corresponding with the land mask, area (m^2), longitude, and latitude
        # at the location of the variable.  The fields correspond to the same
        # location on the Arakawa C-grid as the variable.  For example, 'lon'
        # corresponds to the longitude where variable is situated on the
        # Arakawa C-grid.  To make the location of these auxillary fields
        # unambiguous, a suffix is added based on where on the Arakawa C-grid
        # it is locations.
            
        # By default, we do not keep the redundant land mask and cell area
        # information.
        if keep_landmask_and_area:
            print('Keeping land mask and area!\n')
            if 'land' in ds.variables:
                new_land_name = 'land_' + var_type
                ds = ds.rename({'land':new_land_name})
            if 'area' in ds.variables:
                new_area_name = 'area_' + var_type
                ds = ds.rename({'area':new_area_name})

        # drop these two variables 
        ds=ds.drop(['land', 'area'])
        
        if 'thic' in ds.variables:
            ds = ds.drop(['thic'])
            
        if 'lon' in ds.variables:
            new_lon_name = 'lon_' + var_type
            ds = ds.rename({'lon':new_lon_name})
        if 'lat' in ds.variables:
            new_lon_name = 'lat_' + var_type
            ds = ds.rename({'lat':new_lon_name})
                    

    else:
        # we do have a GRID tile file which requires special treatment.
        
        # First, save the attributes (metadata) of the Dataset.  
        # We'll need it later.
        orig_attrs = ds.attrs

        # find the number of vertical levels (ECCO v4 has 50 levels)
        nk = len(ds.RF)
        
       
        
        # Now give the Dataset helpful coordinate dimension
        # names instead of using 'i1','i2','i3'.
        
        # Start with coordinates for grid parameters at the grid cell "top":
        # (1) the depth of the cell top (RF) and (2) cell center distance (DRC)
        # -- step 1, create a new Dataset with just these two parameters
        GRID_ZT = xr.merge([ds.RF, ds.DRC])
        # -- step 2, rename the vertical coordinate to from 'i1' to 'k_g'
        GRID_ZT = GRID_ZT.rename({'i1':'k_g'})

        # Now coordinates for vertical grid parameters at the cell center 
        # -- step 1, create a new temporary Dataset         
        GRID_ZC = xr.merge([ds.RC, ds.DRF])
        # -- step 2, give proper coordinate names
        GRID_ZC = GRID_ZC.rename({'i1':'k'})

        # Now change the coordinates for grid parameters that are 
        # situated on the 'c', 'g', 'u' and 'v points
        # -- step 1, create a new temporary Dataset with params on 'G' points
        GRID_G = ds[['XG', 'YG','RAZ']]
        # -- step 2, give proper coordinate names
        GRID_G= GRID_G.rename({'i2':'j_g', 'i3':'i_g'})

        # -- step 3, create a new temporary Dataset with params on 'U' points
        GRID_U = ds[['DXC','DYG','hFacW']]
        # -- step 4, give proper coordinate names
        GRID_U= GRID_U.rename({'i1':'k','i2':'j', 'i3':'i_g'})

        
        # -- step 5, create a new temporary Dataset with params on 'V' points
        GRID_V = ds[['DYC','DXG','hFacS']]
        # -- step 6, give proper coordinate names
        GRID_V= GRID_V.rename({'i1':'k','i2':'j_g', 'i3':'i'})
        
        # -- step 7, create a new temporary Dataset with params on 'C' points
        GRID_C = ds[['XC','YC','RAC','Depth','AngleCS','AngleSN','hFacC']]
        # -- step 8, give proper coordinate names
        GRID_C = GRID_C.rename({'i1':'k','i2':'j', 'i3':'i'})        

        
        # Now create a new Dataset for the vertical grid parameter
        # RB: the depth (in meters) of the bottom of the grid cell 
        # -- step 1, create an empty array of length equal to the number of
        #            vertical levels
        #tmp = np.zeros(nk)
        # -- step 2, copy over all the depth coordinates of the grid cell top
        #            starting from k=2 (the "top" of the second grid cell
        #            is the "bottom " of the first grid cell)
        #tmp[0:-1] = GRID_ZT.RF[1:]
        # -- step 3, calculate and add the depth of the deepest grid cell
        #            the depth of the deepest grid cell is the depth of the 
        #            second from the deepest grid cell plus the height of the
        #            the last grid cell
        #tmp[-1] = tmp[-2]-ds.DRF[-1]
        # -- step 4, create a new Dataset with this new array and give it a 
        #            vertical coordinate index 'k_l' with values [1., 2... nk].
        #GRID_ZB = xr.Dataset({'RB': (['k_l'], tmp)},
        #                     coords={'k_l':np.arange(1,nk+1)})
        # -- step 5, give the variable the same attributes as RF (depth in m)
        #GRID_ZB.RB.attrs['long_name'] = 'depth of the deepest grid cell'
        #GRID_ZB.RB.attrs['units'] = 'm'

        # next we'll make land/wet masks.  these will prove very useful
        # later.  
        land_v = deepcopy(GRID_V.hFacS)
        land_v.name = 'land_v'
        land_v[:] = np.ceil(land_v.values)
        land_v.attrs['long_name'] = 'land mask on v points'
        
        land_u = deepcopy(GRID_U.hFacW)
        land_u.name = 'land_u'
        land_u[:] = np.ceil(land_u.values)
        land_u.attrs['long_name'] = 'land mask on u points'
        
        land_c = deepcopy(GRID_C.hFacC)
        land_c.name = 'land_c'
        land_c[:] = np.ceil(land_c.values)
        land_c.attrs['long_name'] = 'land mask on c points'

        # now merge our new and improved Datasets together
        ds = xr.merge([GRID_C, land_c, GRID_G, GRID_U, land_u, GRID_V, land_v,
                       GRID_ZT, GRID_ZC]) #, GRID_ZB])

        # give the new merged grid dataset the original attributes
        # (metadata description)
        ds.attrs =  orig_attrs

        # finally, give more useful descriptions of some vertical grid parameters
        # and give the coordinates more meaningful names
        ds.RC.attrs['long_name'] = 'depth of grid cell center'        
        ds.RF.attrs['long_name'] = 'depth of grid cell top'
        # ds.RB.attrs['long_name'] = 'depth of grid cell bottom'                    
        ds.DRF.attrs['long_name'] = 'vertical distance between the grid cell top and bottom'
        ds.DRC.attrs['long_name'] = 'vertical distance between grid cell centers, starting from the ocean surface'
        
      
        
    # loop through each Dataset variable (which, for some reason also includes 
    # coordinates) and give each rotatable variable (dim >= 2) an attribute
    # 'grid_layout' = 'original llc' to indicate that the tile has not yet
    # been rotated

    for key, value in ds.variables.iteritems():
        if key not in ds.coords and len(ds[key].shape) >= 2:
            ds.variables[key].attrs['grid_layout'] = 'original llc'


    
    #%%
    return ds
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
