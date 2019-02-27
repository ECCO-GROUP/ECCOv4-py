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

from  pyproj import Proj, transform
import matplotlib.path as mpath

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.util as cu


# Performs cartographic transformations and geodetic computations.

# The Proj class can convert from geographic (longitude,latitude) to native 
# map projection (x,y) coordinates and vice versa, or from one map projection 
# coordinate system directly to another.
# https://pypi.python.org/pypi/pyproj?
#
import pyresample as pr

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    



def plot_tiles_proj(lons, lats, data, 
                    user_lat_0 = 45, 
                    projection_type = 'robin', 
                    plot_type = 'pcolor', 
                    user_lon_0 = 0,
                    background_type = 'fc', 
                    show_cbar_label = False, 
                    show_colorbar = False, 
                    cbar_label = '',
                    bound_lat = 50, 
                    num_levels = 20, 
                    cmap='jet', 
                    dx=.25, 
                    dy=.25,
                    show_grid_lines = True,
                    **kwargs):
    
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
    
    if type(data) == xr.core.dataarray.DataArray:
        data = data.values


    elif type(data) != np.ndarray:
        print 'data must be either a DataArray or ndarray type \n'
        print 'found type ', type(data)

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

    #%%
    # the number of degrees spanned in part A and part B
    num_deg_A =  int((A_right_limit - A_left_limit)/dx)
    num_deg_B =  int((B_right_limit - B_left_limit)/dx)

    print num_deg_A, type(num_deg_A)
    print num_deg_B
    
    # We will interpolate the data to the new grid.  Store the longitudes to
    # interpolate to for part A and part B
    lon_tmp_d = dict()
    if num_deg_A > 0:
        lon_tmp_d['A'] = np.linspace(A_left_limit, A_right_limit, num_deg_A)
        
    if num_deg_B > 0:
       lon_tmp_d['B'] = np.linspace(B_left_limit, B_right_limit, num_deg_B)

    #%%


    print ('projection type ', projection_type)
    if projection_type == 'cyl':
        ax = plt.subplot(1,1,1, \
                         projection = ccrs.LambertCylindrical())
    elif projection_type == 'robin':    
        ax = plt.subplot(1,1,1, \
                         projection = ccrs.Robinson(central_longitude=user_lon_0))
    elif projection_type == 'ortho':
        ax = plt.subplot(1,1,1, \
                         projection = ccrs.Orthographic(central_longitude=user_lon_0, \
                                                        central_latitude=user_lat_0))
    elif projection_type == 'stereo':    
        if bound_lat > 0:
            ax = plt.subplot(1,1,1, projection =ccrs.NorthPolarStereo())
        else:
            ax = plt.subplot(1,1,1, projection =ccrs.SouthPolarStereo())
    else:
        raise ValueError('projection type must be either "cyl", "robin", or "stereo"')
        print 'found ', projection_type
    

    # first define the lat lon points of the original data
    orig_grid = pr.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)
    
    #%%
    # the latitudes to which we will we interpolate
    lat_tmp = np.linspace(-89.5, 89.5, int(180.0/dy))
    print lat_tmp
    
    f = plt.gcf()
    #%%
    # loop through both parts (if they exist), do interpolation and plot
    for key, lon_tmp in lon_tmp_d.iteritems():
        print key
        new_grid_lon, new_grid_lat = np.meshgrid(lon_tmp, lat_tmp)
    
        
        # define the lat lon points of the two parts. 
        new_grid  = pr.geometry.GridDefinition(lons=new_grid_lon, 
                                               lats=new_grid_lat)
   
        data_latlon_projection = \
            pr.kd_tree.resample_nearest(orig_grid, data, new_grid, 
                                        radius_of_influence=100000, 
                                        fill_value=None) 

        if plot_type == 'pcolor':
            if (type(ax) == ccrs.NorthPolarStereo) or \
            (type(ax) == ccrs.NorthPolarStereo) :
                p, gl, cbar = plot_pcolormesh_polar_stereographic(new_grid_lon,
                                                                  new_grid_lat, 
                                                              data_latlon_projection,
                                                    4326, bound_lat, cmin, cmax, ax,
                                                    show_colorbar, circle_boundary=True,
                                                    cmap=cmap, draw_labels=False)
            else:
                p, gl, cbar = plot_pcolormesh_global(new_grid_lon,new_grid_lat, 
                                                     data_latlon_projection,
                                       4326, cmin, cmax, ax,
                                       show_colorbar=show_colorbar,
                                       cmap=cmap, draw_labels=False)
        f.show()
        raw_input("Press Enter to continue...")

    #%%
    ax= plt.gca()


    return f, ax, p
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    

def plot_pcolormesh_polar_stereographic(xx,yy, data, 
                                        data_projection_code, \
                                        lat_lim, 
                                        cmin, cmax, ax, \
                                        show_colorbar=False, \
                                        circle_boundary = False, \
                                        cmap='jet',
                                        draw_labels = False):

    if isinstance(ax.projection, ccrs.NorthPolarStereo):
        ax.set_extent([-180, 180, lat_lim, 90], ccrs.PlateCarree())
    elif isinstance(ax.projection, ccrs.SouthPolarStereo):
        ax.set_extent([-180, 180, -90, lat_lim], ccrs.PlateCarree())
    else:
        print 'ax must be either ccrs.NorthPolarStereo or ccrs.SouthPolarStereo'

    if circle_boundary:
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=draw_labels,
                  linewidth=1, color='black', alpha=0.5, linestyle='--')

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs=ccrs.epsg(data_projection_code)
    
    p = ax.pcolormesh(xx, yy, data, transform=data_crs, \
                    vmin=cmin, vmax=cmax, cmap=cmap)
    
    ax.add_feature(cfeature.LAND)
    ax.coastlines('110m', linewidth=0.8)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
    
    return p, gl, cbar

#%%    

def plot_pcolormesh_global(xx,yy, data, 
                           data_projection_code,
                           cmin, cmax, ax, 
                           show_colorbar=False, 
                           cmap='jet', draw_labels = False):

    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  linewidth=1, color='black', draw_labels = draw_labels,
                  alpha=0.5, linestyle='--')

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs =ccrs.epsg(data_projection_code)
        
    p = ax.pcolormesh(xx, yy, data, transform=data_crs, 
                      vmin=cmin, vmax=cmax, cmap=cmap)
    
    ax.coastlines('110m', linewidth=0.8)
    ax.add_feature(cfeature.LAND)

    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
    
    return p, gl, cbar
    

#%%
def plot_contourf_global(xx,yy, data, 
                         data_projection_code, 
                         levels,
                         cmin, cmax, ax, 
                         show_colorbar=False, 
                         cmap='jet',
                         draw_labels = False):

    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
              linewidth=1, color='black', draw_labels = draw_labels,
              alpha=0.5, linestyle='--')

    p = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='white', alpha=0.5, linestyle='--')

    if data_projection_code == 4326: # lat lon does nneed to be projected
        data_crs =  ccrs.PlateCarree()
    else:
        data_crs =  ccrs.epsg(data_projection_code)

    p = ax.contourf(xx, yy, data, levels, transform=data_crs,  \
                 vmin=cmin, vmax=cmax, cmap=cmap)
    
    ax.coastlines('110m', linewidth=0.8)
    ax.add_feature(cfeature.LAND)
    
    cbar = []
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(cmin,cmax))
        sm._A = []
        plt.colorbar(sm,ax=ax)

    return p, gl, cbar
        