#!/usr/bin/env python2
# -*- coding: utf-8 -*-

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
import scipy.interpolate as interpolate
import math

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def resample_to_latlon_nearest(orig_lons, orig_lats, orig_field,
                     new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,
                     new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
                     nprocs_user) :



    #%%
    if type(orig_lats) == xr.core.dataarray.DataArray:
        orig_lons_1d = \
            orig_lons.values.reshape(np.product(orig_lons.values.shape))
        orig_lats_1d = \
            orig_lats.values.reshape(np.product(orig_lats.values.shape))

    elif type(orig_lats) == np.ndarray:
        orig_lats_1d = orig_lats.reshape(np.product(orig_lats.shape))
        orig_lons_1d = orig_lons.reshape(np.product(orig_lons.shape))
    else:
        print 'orig_lons and orig_lats variable either a DataArray or numpy.ndarray'
        print 'orig_lons found type ', type(orig_lons)
        print 'orig_lats found type ', type(orig_lats)
        return

    if type(orig_field) == xr.core.dataarray.DataArray:
        orig_field = orig_field.values
    elif type(orig_field) != np.ndarray:
        print 'orig_fieldmust be either a DataArray or ndarray type \n'
        print 'found type ', type(orig_field)
        return



    # prepare for the nearest neighbor mapping

    # first define the lat lon points of the original data
    orig_grid = pr.geometry.SwathDefinition(lons=orig_lons_1d,
                                            lats=orig_lats_1d)

    # the latitudes to which we will we interpolate

    # the latitudes to which we will we interpolate
    lat_tmp = (np.arange(new_grid_min_lat, new_grid_max_lat,
                         new_grid_delta_lat))
    lon_tmp = (np.arange(new_grid_min_lon, new_grid_max_lon,
                         new_grid_delta_lon))

    new_grid_lon, new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)

    # define the lat lon points of the two parts.
    new_grid  = pr.geometry.GridDefinition(lons=new_grid_lon,
                                           lats=new_grid_lat)


    data_latlon_projection = \
            pr.kd_tree.resample_nearest(orig_grid, orig_field, new_grid,
                                        radius_of_influence=1000000,
                                        fill_value=None,
                                        nprocs=nprocs_user)

    return new_grid_lon, new_grid_lat, data_latlon_projection
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
