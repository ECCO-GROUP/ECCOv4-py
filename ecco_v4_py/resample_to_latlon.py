#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import division,print_function
import numpy as np
import matplotlib.pylab as plt
import xarray as xr

# The Proj class can convert from geographic (longitude,latitude) to native
# map projection (x,y) coordinates and vice versa, or from one map projection
# coordinate system directly to another.
# https://pypi.python.org/pypi/pyproj?
#
import pyresample as pr

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def resample_to_latlon(orig_lons, orig_lats, orig_field,
                       new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,
                       new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
                       nprocs_user=1, radius_of_influence = 100000, 
                       fill_value = None, mapping_method = 'bin_average') :

    #%%
    if type(orig_lats) == xr.core.dataarray.DataArray:
        orig_lons_1d = orig_lons.values.ravel()
        orig_lats_1d = orig_lats.values.ravel()
        
    elif type(orig_lats) == np.ndarray:
        orig_lats_1d = orig_lats.ravel()
        orig_lons_1d = orig_lons.ravel()
    else:
        raise TypeError('orig_lons and orig_lats variable either a DataArray or numpy.ndarray. \n'
                'Found type(orig_lons) = %s and type(orig_lats) = %s' % 
                (type(orig_lons), type(orig_lats)))

    if type(orig_field) == xr.core.dataarray.DataArray:
        orig_field = orig_field.values
    elif type(orig_field) != np.ndarray and \
         type(orig_field) != np.ma.core.MaskedArray :
        raise TypeError('orig_field must be a type of DataArray, ndarray, or MaskedArray. \n' 
                'Found type(orig_field) = %s' % type(orig_field))

    # prepare for the nearest neighbor mapping

    # first define the lat lon points of the original data
    orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d,
                                            lats=orig_lats_1d)

   # the latitudes to which we will we interpolate
    num_lats = (new_grid_max_lat - new_grid_min_lat) / new_grid_delta_lat + 1
    num_lons = (new_grid_max_lon - new_grid_min_lon) / new_grid_delta_lon + 1

    if (num_lats > 0) and (num_lons > 0):
        # linspace is preferred when using floats!
        lat_tmp = np.linspace(new_grid_min_lat, new_grid_max_lat, num=num_lats)
        lon_tmp = np.linspace(new_grid_min_lon, new_grid_max_lon, num=num_lons)

        new_grid_lon, new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)

        # define the lat lon points of the two parts.
        new_grid  = pr.geometry.GridDefinition(lons=new_grid_lon,
                                               lats=new_grid_lat)

        if mapping_method == 'nearest_neighbor':
            data_latlon_projection = \
                    pr.kd_tree.resample_nearest(orig_grid, orig_field, new_grid,
                                                radius_of_influence=radius_of_influence,
                                                fill_value=None,
                                                nprocs=nprocs_user)
        elif mapping_method == 'bin_average':
            wf = lambda r: 1
        
            data_latlon_projection = \
                    pr.kd_tree.resample_custom(orig_grid, orig_field, new_grid,
                                                radius_of_influence=radius_of_influence,
                                                weight_funcs = wf,
                                                fill_value=None,
                                                nprocs=nprocs_user)
        else:
            raise ValueError('mapping_method must be nearest_neighbor or bin_average. \n'
                    'Found mapping_method = %s ' % mapping_method)

    else:
        raise ValueError('Number of lat and lon points to interpolate to must be > 0. \n'
                'Found num_lats = %d, num lons = %d' % (num_lats,num_lons))

    return new_grid_lon, new_grid_lat, data_latlon_projection
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

